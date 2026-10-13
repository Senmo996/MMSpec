"""Evaluation adapter for the existing SAM + ViSkip implementation.

The visual-attention rule and accept-reset gate are inherited unchanged. This
adapter only enforces the paper's generation budget/EOS policy, disables the
legacy repetition-based early exit, and provides a true autoregressive control.
"""

import torch

from method.sam_vskip.spec_model import SpecModel as SAMViSkip
from method.vispec.kv_cache import initialize_past_key_values
from method.vispec.spec_model_ours import _collect_stop_token_ids
from method.vispec_vskip.spec_model import SpecModel as ViSpecViSkip


def bounded_draft_length(raw_length, generated, max_new_tokens, max_draft_tokens):
    return min(raw_length, max_draft_tokens, max(max_new_tokens - generated - 1, 0))


def trim_acceptance(accepted, kept_tokens):
    """Account for committed draft tokens after first-EOS canonicalization.

The initial prefill supplies one token; acceptance records cover subsequent
verification calls. An EOS inside an accepted draft may omit the bonus token.
"""
    remaining = max(kept_tokens - 1, 0)
    result = []
    for length in accepted:
        if remaining <= 0:
            break
        result.append(min(int(length), remaining))
        remaining -= min(int(length) + 1, remaining)
    if remaining:
        raise RuntimeError("acceptance records do not cover the generated tokens")
    return result


class PaperSAMViSkip(SAMViSkip):
    # Reuse the existing standalone AR path; it never constructs SAM drafts.
    _collect_prefill_kwargs = ViSpecViSkip._collect_prefill_kwargs
    _sample_token = ViSpecViSkip._sample_token

    @staticmethod
    def _find_consecutive_ngram_repeat(*args, **kwargs):
        return 0

    def prepare_cache(self, required_tokens):
        """Allocate the same reusable KV workspace for both paired policies."""
        if not hasattr(self, "past_key_values"):
            values = initialize_past_key_values(self.base_model)
            self.past_key_values, self.past_key_values_data, self.current_length_data = values
        for layer in self.past_key_values:
            for cache in layer:
                cache._ensure_capacity(required_tokens)
        self.current_length_data.zero_()

    def _visual_attention_ratio(self, attentions, visual_key_mask, query_index=-1):
        last = attentions[-1] if isinstance(attentions, (list, tuple)) and attentions else attentions
        if last is None or not isinstance(last, torch.Tensor):
            raise RuntimeError("ViSkip requested attention weights but the backend returned none")
        ratio = super()._visual_attention_ratio(attentions, visual_key_mask, query_index)
        self.paper_stats["visual_probes"] += 1
        self.paper_stats["visual_ratio_sum"] += ratio
        if ratio >= self.paper_threshold:
            self.paper_stats["visual_gate_hits"] += 1
        return ratio

    @torch.inference_mode()
    def paper_generate(self, policy, input_ids, max_new_tokens=200,
                       max_draft_tokens=40, visual_attn_threshold=0.35, **kwargs):
        if policy not in ("ar", "sam_vskip"):
            raise ValueError(f"unsupported policy: {policy}")
        if input_ids.shape[0] != 1:
            raise ValueError("the paired paper evaluation requires batch size one")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        prompt_length = int(input_ids.shape[1])
        max_length = prompt_length + max_new_tokens + self.total_tokens + 10
        self.paper_threshold = float(visual_attn_threshold)
        self.paper_stats = dict(visual_probes=0, visual_gate_hits=0,
                                visual_ratio_sum=0., draft_lookups=0, draft_tokens=0)
        if policy == "ar":
            output, n_new, _, accepted = ViSpecViSkip._target_only_generate(
                self, input_ids=input_ids, max_new_tokens=max_new_tokens,
                max_length=max_length, temperature=0., log=True,
                return_acceptance_len=True, **kwargs)
            accepted = accepted[1:]  # AR helper includes its prefill token.
        else:
            original_lookup = self.draft.lookup

            def limited_lookup(start_token):
                kind, tokens, buffers = original_lookup(start_token)
                generated = max(int(self.current_length_data[0]) + 1 - prompt_length, 0)
                count = bounded_draft_length(tokens.numel(), generated,
                                             max_new_tokens, max_draft_tokens)
                self.paper_stats["draft_lookups"] += 1
                self.paper_stats["draft_tokens"] += count
                return kind, tokens[:count], buffers

            self.draft.lookup = limited_lookup
            try:
                output, n_new, _, accepted = super().specgenerate(
                    input_ids=input_ids, max_new_tokens=max_new_tokens,
                    max_length=max_length, temperature=0., log=True,
                    return_acceptance_len=True, skip_when_visual=True,
                    visual_attn_threshold=visual_attn_threshold,
                    visual_probe_only=False, **kwargs)
            finally:
                self.draft.lookup = original_lookup

        generated = output[0, prompt_length:]
        keep = min(int(generated.numel()), max_new_tokens)
        stop_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        for index, token in enumerate(generated[:keep].tolist()):
            if int(token) in stop_ids:
                keep = index + 1
                break
        if int(n_new) > max_new_tokens:
            raise RuntimeError("generation exceeded the guarded token budget")
        physical_iterations = len(accepted)
        accepted = trim_acceptance(accepted, keep)
        return (output[:, :prompt_length + keep], keep, accepted,
                {**self.paper_stats, "verification_iterations": physical_iterations,
                 "canonicalized_tokens": int(n_new) - keep})
