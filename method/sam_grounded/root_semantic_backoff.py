"""Request-local, causal retrieval for an empty lexical root.

Values are existing Top-K prediction rows. A row's key is the final hidden
state of its *predecessor*, not the hidden state that produced that row.
Rows are eligible only while at least one current U/C/G alias retains them;
the bounded FIFO index never changes those lexical dictionaries.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F


def predecessor_keys(hidden, parent_indices, root_predecessor=None):
    """Gather prefix states; -1 means the already verified root predecessor.

    For prefill, a missing predecessor (position zero) is represented by zero
    and must not be inserted. For a packed tree, parent indices are topology
    parents, never simply the previous flattened row.
    """
    indices = torch.as_tensor(parent_indices, device=hidden.device, dtype=torch.long)
    keys = hidden.index_select(0, indices.clamp_min(0)).clone()
    missing = indices.lt(0)
    keys[missing] = 0 if root_predecessor is None else root_predecessor
    return keys


class RootBackoffMemory:
    def __init__(self, mode="cosine", capacity=2048, neighbors=3, top_k=8):
        if mode not in ("cosine", "recent"):
            raise ValueError("root backoff mode must be cosine or recent")
        if capacity < 1 or not 1 <= neighbors <= 3 or top_k < 1:
            raise ValueError("invalid root backoff capacity, neighbors, or Top-K")
        self.mode, self.capacity = mode, int(capacity)
        self.neighbors, self.top_k = int(neighbors), int(top_k)
        self.entries = OrderedDict()
        self.owners = {}
        self.free = list(range(self.capacity - 1, -1, -1))
        self.keys = None
        self.active = None
        self.serial = 0
        self.inserted = self.evicted = self.peak_rows = 0

    def _remove(self, slot, evicted=False):
        entry = self.entries.pop(slot)
        for alias in entry[2]:
            self.owners.pop(alias, None)
        if evicted:
            self.evicted += 1
        self.free.append(slot)

    def update(self, rows, scores, aliases, keys=None):
        if not (len(rows) == len(scores) == len(aliases)):
            raise ValueError("unaligned semantic values and lexical aliases")
        if self.mode == "cosine":
            if keys is None or keys.ndim != 2 or len(keys) != len(rows):
                raise ValueError("cosine retrieval requires one predecessor key per row")
            if self.keys is None:
                self.keys = torch.zeros((self.capacity, keys.shape[-1]), device=keys.device, dtype=torch.float32)
                self.active = torch.zeros(self.capacity, device=keys.device, dtype=torch.bool)
        slots, selected = [], []
        for i, row_aliases in enumerate(aliases):
            row_aliases = set(row_aliases)
            if not row_aliases:
                continue
            for alias in row_aliases:
                previous = self.owners.pop(alias, None)
                if previous is not None:
                    self.entries[previous][2].discard(alias)
                    if not self.entries[previous][2]:
                        self._remove(previous)
            if not self.free:
                self._remove(next(iter(self.entries)), evicted=True)
            slot = self.free.pop()
            self.entries[slot] = (list(rows[i]), list(scores[i]), row_aliases, self.serial)
            self.serial += 1
            self.owners.update((alias, slot) for alias in row_aliases)
            slots.append(slot)
            selected.append(i)
            self.inserted += 1
        if self.mode == "cosine" and slots:
            # Reused slots may occur within a batch. Keep the final assignment
            # explicitly; repeated CUDA scatter destinations are not ordered.
            last = dict(zip(slots, selected))
            ids = torch.tensor(list(last), device=self.keys.device, dtype=torch.long)
            source = torch.tensor(list(last.values()), device=keys.device, dtype=torch.long)
            self.keys.index_copy_(0, ids, F.normalize(keys.index_select(0, source).float(), dim=-1))
            self.active.zero_()
            self.active[list(self.entries)] = True
        self.peak_rows = max(self.peak_rows, len(self.entries))

    def retrieve(self, query=None):
        count = min(self.neighbors, len(self.entries))
        if not count:
            return [], []
        if self.mode == "recent":
            selected = list(reversed(self.entries))[:count]
        else:
            if query is None:
                raise ValueError("a verified predecessor query is required")
            similarities = torch.mv(self.keys, F.normalize(query.float(), dim=-1))
            similarities.masked_fill_(~self.active, -torch.inf)
            selected = torch.topk(similarities, count, sorted=True).indices.tolist()
        fused = {}
        for weight, slot in zip((.7, .2, .1), selected):
            row, probabilities, _, _ = self.entries[slot]
            for token, probability in zip(row, probabilities):
                fused[token] = fused.get(token, 0.) + weight * probability
        candidates = sorted(fused, key=lambda token: -fused[token])[:self.top_k]
        total = sum(fused[token] for token in candidates) or 1.
        return candidates, [fused[token] / total for token in candidates]

    def stats(self):
        return dict(backoff_index_rows=len(self.entries), backoff_index_peak_rows=self.peak_rows,
                    backoff_index_inserted=self.inserted, backoff_index_evicted=self.evicted,
                    backoff_index_key_bytes=0 if self.keys is None else self.keys.numel() * self.keys.element_size())
