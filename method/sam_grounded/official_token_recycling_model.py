"""Token Recycling (Luo et al., ACL 2025) on the shared multimodal verifier.

Follows the official implementation (github.com/Luowaterbi/TokenRecycling,
commit 1b4c05c, ``evaluation/inference_recycling.py``):

* a zero-initialised ``|V| x 8`` adjacency matrix that is never reset between
  requests (hot start); the evaluators clear it only after warmup and at
  benchmark boundaries, matching the RECAP persistent-memory protocol;
* the static, imbalanced tree template "2.2.2" (80 nodes, depth 6), filled by
  a layer-wise gather from the matrix without validity checks, duplicate
  removal, or token filtering;
* one tree-attention verification per step, the longest matching root-to-leaf
  path plus one target token committed, and the Top-8 predictions of every
  verified node written back to the matrix.

Prefill differs only in plumbing: the shared multimodal prefill covers the
whole prompt and emits the first token greedily (as for every other method),
whereas the official text-only code verifies the last prompt token inside the
first tree.  Prefill logits are not written to the matrix, as in the original.
"""

import inspect
from typing import List

import torch

from method.vispec.spec_model_ours import _collect_stop_token_ids, _has_stop_token

from .spec_model import (
    SUPPORTED_MULTIMODAL_ARCHITECTURES,
    resolve_generation_max_length,
    resolve_image_token_id,
)
from .tree_recycling_model import TreeRecyclingSpecModel


OFFICIAL_POLICY = "token-recycling-official"
OFFICIAL_TOP_K = 8
# Verbatim copy of choose_tree_template("2.2.2") from the official repository.
TREE_TEMPLATE_2_2_2 = [
    [0], [1], [2], [3], [4], [5], [6], [7],
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [1, 0],
    [1, 1], [1, 2], [1, 3], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [4, 0],
    [5, 0], [6, 0], [7, 0],
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
    [0, 0, 6], [0, 0, 7], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0],
    [0, 2, 1], [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 6, 0], [0, 7, 0],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
    [5, 0, 0],
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4],
    [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 2, 0], [0, 0, 3, 0], [0, 0, 4, 0],
    [0, 1, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0],
    [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0],
    [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0],
]
_PARENT_PARAMETERS = frozenset(
    inspect.signature(TreeRecyclingSpecModel.specgenerate).parameters
)


def build_template_topology(template, device):
    """Precompute the official template's mask, depths, gather plan and paths.

    Node 0 is the root.  ``search_paths`` holds one root-to-leaf index path per
    leaf, padded with -1, exactly as ``prepare_data`` in the official code.
    """

    num_nodes = len(template) + 1
    index_of = {tuple(path): node for node, path in enumerate(template, start=1)}
    if len(index_of) != len(template):
        raise ValueError("tree template contains duplicate paths")
    mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    mask[:, 0] = True
    depths = [0] * num_nodes
    parents = [-1] * num_nodes
    ranks = [-1] * num_nodes
    for node, path in enumerate(template, start=1):
        if len(path) > 1 and tuple(path[:-1]) not in index_of:
            raise ValueError(f"tree template node {path} has no parent")
        if node > 1 and len(path) < len(template[node - 2]):
            raise ValueError("tree template must be ordered by depth")
        depths[node] = len(path)
        parents[node] = index_of[tuple(path[:-1])] if len(path) > 1 else 0
        ranks[node] = int(path[-1])
        ancestor = node
        while ancestor > 0:
            mask[node, ancestor] = True
            ancestor = parents[ancestor]
    if max(ranks) >= OFFICIAL_TOP_K:
        raise ValueError("tree template rank exceeds the matrix width")

    is_parent = set(parents[1:])
    leaf_paths: List[List[int]] = []
    for node in range(1, num_nodes):
        if node in is_parent:
            continue
        path = [node]
        while parents[path[-1]] > 0:
            path.append(parents[path[-1]])
        leaf_paths.append([0] + list(reversed(path)))
    pad_len = max(depths) + 1
    search_paths = [path + [-1] * (pad_len - len(path)) for path in leaf_paths]

    layers = []
    start = 1
    while start < num_nodes:
        end = start
        while end < num_nodes and depths[end] == depths[start]:
            end += 1
        layers.append((start, end))
        start = end
    return {
        "num_nodes": num_nodes,
        "mask": mask.to(device)[None, None],
        "depths": torch.tensor(depths, dtype=torch.long, device=device),
        "parents": torch.tensor(parents, dtype=torch.long, device=device),
        "ranks": torch.tensor(ranks, dtype=torch.long, device=device),
        "layers": layers,
        "search_paths": torch.tensor(search_paths, dtype=torch.long, device=device),
    }


class OfficialTokenRecyclingSpecModel(TreeRecyclingSpecModel):
    """Official Token Recycling drafts; other policies use the parent class."""

    def reset_persistent_recycling_cache(self) -> None:
        super().reset_persistent_recycling_cache()
        self._official_tr_matrix = None

    def _official_matrix(self, vocab_size: int, device) -> torch.Tensor:
        matrix = getattr(self, "_official_tr_matrix", None)
        if matrix is None or matrix.shape[0] != vocab_size or matrix.device != device:
            matrix = torch.zeros(
                (vocab_size, OFFICIAL_TOP_K), dtype=torch.long, device=device
            )
            self._official_tr_matrix = matrix
        return matrix

    def _official_topology(self, device):
        topology = getattr(self, "_official_tr_topology", None)
        if topology is None or topology["mask"].device != device:
            topology = build_template_topology(TREE_TEMPLATE_2_2_2, device)
            self._official_tr_topology = topology
        return topology

    @torch.no_grad()
    def specgenerate(self, input_ids, draft_policy="visual-anchor", **kwargs):
        if draft_policy != OFFICIAL_POLICY:
            return super().specgenerate(
                input_ids, draft_policy=draft_policy, **kwargs
            )
        options = {k: v for k, v in kwargs.items() if k in _PARENT_PARAMETERS}
        model_kwargs = {
            k: v for k, v in kwargs.items() if k not in _PARENT_PARAMETERS
        }
        return self._official_generate(input_ids, model_kwargs, **options)

    def _official_generate(
        self,
        input_ids,
        model_kwargs,
        temperature=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        inputs_embeds=None,
        return_acceptance_len=False,
        return_decode_time=False,
        return_policy_trace=False,
        **unused_options,
    ):
        del unused_options
        if inputs_embeds is not None:
            raise ValueError("Official Token Recycling expects input_ids")
        if input_ids.shape[0] != 1:
            raise ValueError("Official Token Recycling supports batch size 1")
        if temperature > 1e-5:
            raise NotImplementedError("Official Token Recycling is greedy-only here")
        arch = self.base_model.config.architectures[0]
        if arch not in SUPPORTED_MULTIMODAL_ARCHITECTURES:
            raise NotImplementedError(f"Unsupported architecture {arch}")

        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        prompt_length = int(input_ids.shape[1])
        max_length = resolve_generation_max_length(
            self.base_model.config, prompt_length, max_new_tokens, max_length
        )
        device = input_ids.device
        topology = self._official_topology(device)
        matrix = self._official_matrix(
            int(self.base_model.config.vocab_size), device
        )
        num_nodes = topology["num_nodes"]
        parents = topology["parents"]
        ranks = topology["ranks"]
        search_paths = topology["search_paths"]
        path_valid = search_paths.ge(0)
        safe_paths = search_paths.clamp(min=0)

        if hasattr(self, "tree_recycling_past_key_values"):
            past_key_values = self.tree_recycling_past_key_values
            past_key_values_data = self.tree_recycling_past_key_values_data
            current_length_data = self.tree_recycling_current_length_data
            current_length_data.zero_()
        else:
            from method.vispec.kv_cache import initialize_past_key_values

            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.tree_recycling_past_key_values = past_key_values
            self.tree_recycling_past_key_values_data = past_key_values_data
            self.tree_recycling_current_length_data = current_length_data

        if arch == "Qwen2_5_VLForConditionalGeneration":
            image_token_id = resolve_image_token_id(self.base_model.config)
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
            pixel_values = model_kwargs.get("pixel_values")
            if pixel_values is not None:
                image_embeds = self.base_model.visual(
                    pixel_values.type(self.base_model.visual.dtype),
                    grid_thw=model_kwargs.get("image_grid_thw"),
                )
                image_mask = input_ids.eq(image_token_id)
                if int(image_mask.sum().item()) != int(image_embeds.shape[0]):
                    raise ValueError("Image features and image tokens do not match")
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(inputs_embeds),
                    image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
                )
        init_output = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=True,
            **model_kwargs,
        )

        if hasattr(self.base_model, "language_model"):
            language_model = self.base_model.language_model
            tree_mask_model = getattr(language_model, "model", language_model)
        else:
            tree_mask_model = self.base_model.model

        acceptance_lengths: List[int] = []
        trace = []
        idx = -1
        if max_new_tokens > 0 and input_ids.shape[1] < max_length:
            first = torch.argmax(init_output.logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, first[:, None]], dim=1)
            current_length_data.fill_(input_ids.shape[1] - 1)
        del init_output

        tree_tokens = torch.empty(num_nodes, dtype=torch.long, device=device)
        while True:
            generated = int(input_ids.shape[1] - prompt_length)
            if generated >= max_new_tokens or input_ids.shape[1] >= max_length:
                break
            if _has_stop_token(input_ids[0, prompt_length:], stop_token_ids):
                break
            idx += 1
            remaining = int(max_new_tokens - generated)

            # Layer-wise BFS gather (prepare_data_input_ids in the original).
            tree_tokens[0] = input_ids[0, -1]
            for start, end in topology["layers"]:
                tree_tokens[start:end] = matrix[
                    tree_tokens[parents[start:end]], ranks[start:end]
                ]

            kv_base_len = int(current_length_data[0].item())
            position_ids = topology["depths"] + kv_base_len
            if arch == "Qwen2_5_VLForConditionalGeneration":
                position_ids = position_ids.unsqueeze(0) + self.base_model.rope_deltas
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            else:
                position_ids = position_ids.unsqueeze(0)
            tree_mask_model.tree_mask = topology["mask"]
            try:
                output = self.base_model(
                    input_ids=tree_tokens.unsqueeze(0),
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True,
                )
            finally:
                tree_mask_model.tree_mask = None
            logits = output.logits[0, :num_nodes]
            predictions = torch.argmax(logits, dim=-1)

            # Longest verified root-to-leaf prefix (the official cumprod reward).
            path_inputs = tree_tokens[safe_paths].masked_fill(~path_valid, -100)
            path_outputs = predictions[safe_paths]
            reward = torch.cumprod(
                path_inputs[:, 1:].eq(path_outputs[:, :-1]), dim=-1
            ).sum(dim=-1)
            best = torch.argmax(reward)
            matrix[tree_tokens] = torch.topk(logits, k=OFFICIAL_TOP_K, dim=-1).indices

            accept = min(int(reward[best].item()), max(remaining - 1, 0))
            node_path = search_paths[best, : accept + 1]
            committed = torch.cat(
                [tree_tokens[node_path[1:]], predictions[node_path[-1:]]]
            )[:remaining]
            if accept > 0:
                select_indices = node_path + kv_base_len
                for cache_data in past_key_values_data:
                    selected = cache_data.index_select(
                        -2, select_indices.to(cache_data.device)
                    )
                    cache_data.narrow(-2, kv_base_len, selected.shape[-2]).copy_(
                        selected, non_blocking=True
                    )
            current_length_data.fill_(kv_base_len + accept + 1)
            input_ids = torch.cat([input_ids, committed.unsqueeze(0)], dim=1)
            acceptance_lengths.append(accept)
            if return_policy_trace:
                trace.append(
                    {
                        "iteration": len(trace),
                        "verified_tree_nodes": num_nodes - 1,
                        "remaining_tokens": remaining,
                        "accept_len": accept,
                    }
                )

        generated_ids = input_ids[0, prompt_length:]
        keep_tokens = min(int(generated_ids.numel()), int(max_new_tokens))
        for token_index, token_id in enumerate(generated_ids[:keep_tokens].tolist()):
            if int(token_id) in stop_token_ids:
                keep_tokens = token_index + 1
                break
        input_ids = input_ids[:, : prompt_length + keep_tokens]
        outputs = (input_ids,)
        if log:
            outputs += (keep_tokens, max(idx, 0))
        if return_acceptance_len:
            outputs += (acceptance_lengths,)
        if return_decode_time:
            outputs += (0.0,)
        if return_policy_trace:
            outputs += (trace,)
        return outputs[0] if len(outputs) == 1 else outputs
