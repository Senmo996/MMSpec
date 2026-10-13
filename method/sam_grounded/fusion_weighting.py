"""Prespecified, inference-only source weighting for RECAP ablations."""

import math


ONLINE_RULES = ("fixed", "uniform", "confidence", "prior_confidence", "confidence_gate")
OFFLINE_RULES = ONLINE_RULES + ("prior_soft", "prior_sharp")
PRIORS = {"fixed": (.70, .20, .10, .05),
          "prior_soft": (.55, .30, .15, .05),
          "prior_sharp": (.85, .10, .05, .025)}


def concentration(probabilities):
    """Inverse perplexity of the retained row, not calibrated correctness."""
    total = sum(probabilities)
    if not probabilities or total <= 0 or not math.isfinite(total):
        raise ValueError("concentration requires a nonempty positive row")
    values = [float(p) / total for p in probabilities]
    if any(p < 0 or not math.isfinite(p) for p in values):
        raise ValueError("invalid prediction probabilities")
    return math.exp(sum(p * math.log(p) for p in values if p > 0))


def source_weights(rows, rule):
    if rule not in OFFLINE_RULES:
        raise ValueError(f"unknown fusion rule: {rule}")
    if not rows:
        return []
    if rule in PRIORS:
        prior = PRIORS[rule]
        return [prior[min(i, len(prior) - 1)] for i in range(len(rows))]
    if rule == "uniform":
        return [1.] * len(rows)
    confidence = [concentration(scores) for _, scores in rows]
    if rule == "confidence":
        return confidence
    if rule == "prior_confidence":
        return [c * PRIORS["fixed"][min(i, 3)] for i, c in enumerate(confidence)]
    winner = max(range(len(rows)), key=confidence.__getitem__)
    return [float(i == winner) for i in range(len(rows))]


class FusionRule:
    """Callable accepted by the shared tree builder; counters are optional."""

    def __init__(self, rule, counters=None):
        if rule not in OFFLINE_RULES:
            raise ValueError(rule)
        self.rule, self.counters = rule, counters

    def __call__(self, rows, limit):
        if not rows or limit <= 0:
            return [], []
        if len(rows) == 1:
            if self.counters is not None:
                self.counters["single_source_calls"] = self.counters.get("single_source_calls", 0) + 1
            return rows[0]
        weights = source_weights(rows, self.rule)
        if self.counters is not None:
            stats = self.counters
            stats["multiple_source_calls"] = stats.get("multiple_source_calls", 0) + 1
            normalized = [w / sum(weights) for w in weights]
            key = f"weight_sums_{len(rows)}_sources"
            stats[key] = [a + b for a, b in zip(stats.get(key, [0.] * len(rows)), normalized)]
            count_key = f"calls_{len(rows)}_sources"
            stats[count_key] = stats.get(count_key, 0) + 1
            if self.rule == "confidence_gate":
                winner = weights.index(1.)
                stats["gate_selected_rank_counts"] = [a + int(i == winner) for i, a in
                    enumerate(stats.get("gate_selected_rank_counts", [0] * 4))]
        scores, first_seen = {}, {}
        for weight, (candidates, probabilities) in zip(weights, rows):
            if weight == 0:
                continue
            for token, probability in zip(candidates, probabilities):
                token = int(token)
                scores[token] = scores.get(token, 0.) + weight * float(probability)
                first_seen.setdefault(token, len(first_seen))
        selected = sorted(scores, key=lambda token: (-scores[token], first_seen[token]))[:limit]
        total = sum(scores[token] for token in selected) or 1.
        return selected, [scores[token] / total for token in selected]
