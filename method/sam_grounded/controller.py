"""Training-free controllers for multimodal speculative draft length.

The controller deliberately operates only on signals already produced by the
target-model verification pass.  It therefore needs neither learned weights nor
an additional model forward.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F


TRIGRAM_PLUS4_NODE_BUDGET_POLICIES = frozenset(
    f"context-score-trigram-deeper-wide-plus4-node{budget}"
    for budget in (23, 31, 39, 47, 55)
)

PERSISTENT_FUSION_NODE_BUDGET_POLICIES = frozenset(
    f"context-score-trigram-fusion-persistent-node{budget}-deepest-wide-plus4"
    for budget in (39, 47, 55, 63, 79, 95)
)

PERSISTENT_COMMITTED_POLICIES = frozenset(
    {"context-score-trigram-fusion-persistent-committed-deepest-wide-plus4"}
)

PERSISTENT_STABLE_NODE_BUDGET_POLICIES = frozenset(
    f"context-score-trigram-fusion-persistent-stable-node{budget}-deepest-wide-plus4"
    for budget in (55, 63)
)

PERSISTENT_DEPTH_NODE_BUDGET_POLICIES = frozenset(
    f"context-score-trigram-fusion-persistent-depth{depth}-node{budget}-wide-plus4"
    for depth, budget in (
        (7, 63),
        (8, 55),
        (8, 63),
        (10, 47),
        (10, 55),
        (10, 63),
        (10, 79),
        (10, 95),
    )
)

PERSISTENT_ADAPTIVE_DEPTH_POLICIES = frozenset(
    {
        "context-score-trigram-fusion-persistent-adaptive95-"
        "depth10-wide-plus4"
    }
)

PERSISTENT_OPTIMIZED_DEPTH_POLICIES = frozenset(
    {
        "context-score-trigram-fusion-persistent-contextcal-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-shadow-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-hotpath-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes-hotpath-cpp-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-contextnodes95-hotpath-cpp-"
        "depth10-node95-wide-plus4",
        "context-score-trigram-fusion-persistent-hotpath-cpp-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-hotpath-cpp-"
        "depth10-node79-wide-plus4",
        "context-score-trigram-fusion-persistent-hotpath-cpp-"
        "depth10-node95-wide-plus4",
        "context-score-trigram-fusion-persistent-empirical-hotpath-cpp-"
        "depth14-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-suffix4-hotpath-cpp-"
        "depth10-node63-wide-plus4",
        "context-score-trigram-fusion-persistent-suffix4-visualcache-"
        "hotpath-cpp-depth10-node63-wide-plus4",
    }
)


@dataclass(frozen=True)
class DraftDecision:
    budget: int
    risk: float
    grounding_score: float
    confidence: float
    acceptance_ema: float

    def to_dict(self):
        return asdict(self)


class VisualGroundingCalibrator:
    """Turn prompt hidden states into a calibrated visual-grounding score.

    Visual and textual prompt tokens define two centroids in the target model's
    own hidden space.  A generated-token state is projected onto the direction
    separating those centroids.  The textual and visual prompt medians calibrate
    the projection to approximately [0, 1].
    """

    def __init__(
        self,
        visual_prototype: torch.Tensor,
        text_prototype: torch.Tensor,
        text_center: float,
        visual_center: float,
        eps: float = 1e-5,
    ):
        self.visual_prototype = visual_prototype
        self.text_prototype = text_prototype
        self.text_center = float(text_center)
        self.visual_center = float(visual_center)
        self.eps = float(eps)

    @classmethod
    def from_prompt(
        cls,
        prompt_hidden: torch.Tensor,
        visual_mask: torch.Tensor,
        eps: float = 1e-5,
    ) -> Optional["VisualGroundingCalibrator"]:
        if prompt_hidden.dim() == 3:
            if prompt_hidden.shape[0] != 1:
                raise ValueError("VisualGroundingCalibrator only supports batch size 1")
            prompt_hidden = prompt_hidden[0]
        if prompt_hidden.dim() != 2:
            raise ValueError("prompt_hidden must have shape [sequence, hidden]")

        visual_mask = visual_mask.to(prompt_hidden.device, dtype=torch.bool).view(-1)
        if visual_mask.numel() != prompt_hidden.shape[0]:
            raise ValueError("visual_mask length must match the prompt sequence length")
        text_mask = ~visual_mask
        if not bool(visual_mask.any().item()) or not bool(text_mask.any().item()):
            return None

        normalized = F.normalize(prompt_hidden.detach().float(), dim=-1, eps=eps)
        visual_prototype = F.normalize(
            normalized[visual_mask].mean(dim=0), dim=0, eps=eps
        )
        text_prototype = F.normalize(
            normalized[text_mask].mean(dim=0), dim=0, eps=eps
        )
        direction = visual_prototype - text_prototype
        prompt_projection = normalized @ direction
        text_center = torch.median(prompt_projection[text_mask]).item()
        visual_center = torch.median(prompt_projection[visual_mask]).item()

        # The two class centroids should induce this ordering.  Keep a small,
        # explicit fallback for degenerate layers/prompts rather than allowing
        # an unstable division to drive the policy.
        if visual_center <= text_center + eps:
            midpoint = 0.5 * (visual_center + text_center)
            text_center = midpoint - eps
            visual_center = midpoint + eps

        return cls(
            visual_prototype=visual_prototype,
            text_prototype=text_prototype,
            text_center=text_center,
            visual_center=visual_center,
            eps=eps,
        )

    def score(self, query_hidden: torch.Tensor) -> float:
        return float(self.scores(query_hidden).item())

    def scores(self, query_hidden: torch.Tensor) -> torch.Tensor:
        """Vectorized grounding scores for one or more hidden states."""
        query = query_hidden.detach().float()
        if query.shape[-1] != self.visual_prototype.numel():
            raise ValueError("query hidden size must match grounding prototypes")
        query = F.normalize(query, dim=-1, eps=self.eps)
        projection = query @ (self.visual_prototype - self.text_prototype)
        scale = max(self.visual_center - self.text_center, self.eps)
        return ((projection - self.text_center) / scale).clamp(0.0, 1.0)

    def diagnostics(self):
        return {
            "text_center": self.text_center,
            "visual_center": self.visual_center,
            "separation": self.visual_center - self.text_center,
        }


def confidence_from_logits(logits: torch.Tensor, margin_scale: float = 5.0) -> float:
    """Map greedy top-1/top-2 logit margin to a bounded confidence score."""
    values = torch.topk(logits.detach().float().view(-1), k=2).values
    margin = max(float((values[0] - values[1]).item()), 0.0)
    scale = max(float(margin_scale), 1e-6)
    return float(1.0 - torch.exp(torch.tensor(-margin / scale)).item())


class GroundedDraftController:
    """Choose the next SAM draft cap from target-model state diagnostics."""

    POLICIES = {
        "target",
        "fixed",
        "broad",
        "wide-plus2",
        "rank-prior-wide-plus2",
        "rank-prior-deep-wide-plus2",
        "rank-prior-deeper-wide-plus2",
        "rank-prior-deepest-wide-plus2",
        "score-prior-deep-wide-plus2",
        "score-prior-deeper-wide-plus2",
        "context-score-prior-deeper-wide-plus2",
        "context-score-prior-deeper-wide-plus4",
        "context-score-trigram-deeper-wide-plus4",
        "context-score-trigram-residual2-deeper-wide-plus4",
        "context-score-trigram-residual2-deepest-wide-plus4",
        "context-score-trigram-fusion-deeper-wide-plus4",
        "context-score-trigram-fusion-deepest-wide-plus4",
        "context-score-trigram-fusion55-deepest-wide-plus4",
        "context-score-trigram-fusion-adaptive-deepest-wide-plus4",
        "context-score-trigram-fusion-calibrated-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-global15-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-ngram-deepest-wide-plus4",
        "context-score-trigram-fusion-persistent-ngram-global15-deepest-wide-plus4",
        "context-score-trigram-fusion-bank-deepest-wide-plus4",
        "context-score-trigram-fusion-bank-global15-deepest-wide-plus4",
        "context-score-trigram-fusion-global7-deepest-wide-plus4",
        "context-score-trigram-fusion-global15-deepest-wide-plus4",
        "context-score-trigram-deepest-wide-plus4",
        "context-score-trigram-deeper-wide-plus6",
        "context-score-trigram-deeper-wide-plus8",
        "context-score-fourgram-deeper-wide-plus4",
        "context-score-fourgram-deepest-wide-plus4",
        "context-score-prior-deeper-wide-plus5",
        "context-score-prior-deeper-wide-plus6",
        "context-score-prior-deeper-wide-plus8",
        "context-score-calibrated-deeper-wide-plus2",
        "context-score-prior-deeper-wide-plus2-node55",
        "context-score-prior-deeper-wide-plus2-node47",
        "context-score-adaptive-safe-deeper-wide-plus2",
        "score-prior-deepest-wide-plus2",
        "score-prior-maxdeep-wide-plus2",
        "score-adaptive-safe-deeper-wide-plus2",
        "score-adaptive-deeper-wide-plus2",
        "visual-wide-plus2",
        "visual-wide-plus2-reverse",
        "visual-rootwide-plus2",
        "visual-rootwide-plus2-hst-backoff",
        "visual-lexical-backoff",
        "visual-hst-backoff",
        "visual-hst-backoff-gated",
        "visual-wide-plus2-hst-backoff",
        "visual-wide-plus2-hst-backoff-gated",
        "visual-wide-plus2-vli-backoff",
        "narrow",
        "spine",
        "short",
        "visual-hard",
        "visual-reverse",
        "visual-width",
        "visual-width-reverse",
        "visual-spine",
        "visual-spine-reverse",
        "modal-broad",
        "modal-hard",
        "modal-spine",
        "grounded-backoff",
        "grounded-backoff-reverse",
        "grounded-residual",
        "grounded-residual-reverse",
        "hybrid",
        "visual-hybrid",
        "modal-hybrid",
        "grounded-hybrid",
        "grounded-hybrid-reverse",
        "visual-anchor",
        "visual-soft",
        "visual-accept",
    } | (
        TRIGRAM_PLUS4_NODE_BUDGET_POLICIES
        | PERSISTENT_FUSION_NODE_BUDGET_POLICIES
        | PERSISTENT_COMMITTED_POLICIES
        | PERSISTENT_STABLE_NODE_BUDGET_POLICIES
        | PERSISTENT_DEPTH_NODE_BUDGET_POLICIES
        | PERSISTENT_ADAPTIVE_DEPTH_POLICIES
        | PERSISTENT_OPTIMIZED_DEPTH_POLICIES
    )

    def __init__(
        self,
        policy: str,
        min_draft_tokens: int = 2,
        max_draft_tokens: int = 40,
        visual_threshold: float = 0.55,
        confidence_threshold: float = 0.75,
        visual_gamma: float = 1.0,
        visual_weight: float = 0.7,
        acceptance_weight: float = 0.3,
        acceptance_ema_decay: float = 0.8,
    ):
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown policy {policy!r}; choose from {sorted(self.POLICIES)}")
        if min_draft_tokens < 0 or max_draft_tokens < min_draft_tokens:
            raise ValueError("Require 0 <= min_draft_tokens <= max_draft_tokens")
        if not 0.0 <= acceptance_ema_decay < 1.0:
            raise ValueError("acceptance_ema_decay must be in [0, 1)")

        self.policy = policy
        self.min_draft_tokens = int(min_draft_tokens)
        self.max_draft_tokens = int(max_draft_tokens)
        self.visual_threshold = float(visual_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.visual_gamma = max(float(visual_gamma), 1e-6)
        self.visual_weight = max(float(visual_weight), 0.0)
        self.acceptance_weight = max(float(acceptance_weight), 0.0)
        self.acceptance_ema_decay = float(acceptance_ema_decay)
        self.acceptance_ema = 1.0

    def decide(self, grounding_score: float, confidence: float = 1.0) -> DraftDecision:
        grounding_score = float(max(0.0, min(1.0, grounding_score)))
        confidence = float(max(0.0, min(1.0, confidence)))

        if self.policy == "target":
            budget, risk = 0, 1.0
        elif self.policy in (
            "fixed",
            "broad",
            "wide-plus2",
            "rank-prior-wide-plus2",
            "rank-prior-deep-wide-plus2",
            "rank-prior-deeper-wide-plus2",
            "rank-prior-deepest-wide-plus2",
            "score-prior-deep-wide-plus2",
            "score-prior-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus4",
            "context-score-trigram-deeper-wide-plus4",
            "context-score-trigram-residual2-deeper-wide-plus4",
            "context-score-trigram-residual2-deepest-wide-plus4",
            "context-score-trigram-fusion-deeper-wide-plus4",
            "context-score-trigram-fusion-deepest-wide-plus4",
            "context-score-trigram-fusion55-deepest-wide-plus4",
            "context-score-trigram-fusion-adaptive-deepest-wide-plus4",
            "context-score-trigram-fusion-calibrated-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-ngram-deepest-wide-plus4",
            "context-score-trigram-fusion-persistent-ngram-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-bank-deepest-wide-plus4",
            "context-score-trigram-fusion-bank-global15-deepest-wide-plus4",
            "context-score-trigram-fusion-global7-deepest-wide-plus4",
            "context-score-trigram-fusion-global15-deepest-wide-plus4",
            "context-score-trigram-deepest-wide-plus4",
            "context-score-trigram-deeper-wide-plus6",
            "context-score-trigram-deeper-wide-plus8",
            "context-score-fourgram-deeper-wide-plus4",
            "context-score-fourgram-deepest-wide-plus4",
            "context-score-prior-deeper-wide-plus5",
            "context-score-prior-deeper-wide-plus6",
            "context-score-prior-deeper-wide-plus8",
            "context-score-calibrated-deeper-wide-plus2",
            "context-score-prior-deeper-wide-plus2-node55",
            "context-score-prior-deeper-wide-plus2-node47",
            "context-score-adaptive-safe-deeper-wide-plus2",
            "score-prior-deepest-wide-plus2",
            "score-prior-maxdeep-wide-plus2",
            "score-adaptive-safe-deeper-wide-plus2",
            "score-adaptive-deeper-wide-plus2",
            "visual-wide-plus2",
            "visual-wide-plus2-reverse",
            "visual-rootwide-plus2",
            "visual-rootwide-plus2-hst-backoff",
            "visual-lexical-backoff",
            "visual-hst-backoff",
            "visual-hst-backoff-gated",
            "visual-wide-plus2-hst-backoff",
            "visual-wide-plus2-hst-backoff-gated",
            "visual-wide-plus2-vli-backoff",
            "narrow",
            "spine",
            "modal-broad",
            "grounded-backoff",
            "grounded-backoff-reverse",
            "grounded-residual",
            "grounded-residual-reverse",
            "hybrid",
            "visual-hybrid",
            "modal-hybrid",
            "grounded-hybrid",
            "grounded-hybrid-reverse",
            "visual-width",
            "visual-width-reverse",
        ) or self.policy in (
            TRIGRAM_PLUS4_NODE_BUDGET_POLICIES
            | PERSISTENT_FUSION_NODE_BUDGET_POLICIES
            | PERSISTENT_COMMITTED_POLICIES
            | PERSISTENT_STABLE_NODE_BUDGET_POLICIES
            | PERSISTENT_DEPTH_NODE_BUDGET_POLICIES
            | PERSISTENT_ADAPTIVE_DEPTH_POLICIES
            | PERSISTENT_OPTIMIZED_DEPTH_POLICIES
        ):
            budget, risk = self.max_draft_tokens, 0.0
        elif self.policy == "short":
            budget, risk = self.min_draft_tokens, 1.0
        elif self.policy in ("visual-hard", "modal-hard"):
            risk = grounding_score
            budget = (
                self.min_draft_tokens
                if grounding_score >= self.visual_threshold
                else self.max_draft_tokens
            )
        elif self.policy == "visual-reverse":
            risk = 1.0 - grounding_score
            budget = (
                self.min_draft_tokens
                if grounding_score < self.visual_threshold
                else self.max_draft_tokens
            )
        elif self.policy in (
            "visual-spine",
            "visual-spine-reverse",
            "modal-spine",
        ):
            risk = (
                grounding_score
                if self.policy in ("visual-spine", "modal-spine")
                else 1.0 - grounding_score
            )
            budget = self.max_draft_tokens
        elif self.policy == "visual-anchor":
            # Treat visually aligned, low-confidence predictors as grounded
            # anchors.  Confident visual tokens retain long drafts; ambiguous
            # anchors use a shallow proposal so one bad branch is inexpensive.
            is_anchor = (
                grounding_score >= self.visual_threshold
                and confidence <= self.confidence_threshold
            )
            risk = grounding_score * (1.0 - confidence)
            budget = self.min_draft_tokens if is_anchor else self.max_draft_tokens
        else:
            visual_risk = grounding_score ** self.visual_gamma
            if self.policy == "visual-accept":
                normalizer = self.visual_weight + self.acceptance_weight
                if normalizer <= 0:
                    risk = visual_risk
                else:
                    risk = (
                        self.visual_weight * visual_risk
                        + self.acceptance_weight * (1.0 - self.acceptance_ema)
                    ) / normalizer
            else:
                risk = visual_risk
            span = self.max_draft_tokens - self.min_draft_tokens
            budget = int(round(self.max_draft_tokens - span * risk))
            budget = max(self.min_draft_tokens, min(self.max_draft_tokens, budget))

        return DraftDecision(
            budget=int(budget),
            risk=float(risk),
            grounding_score=grounding_score,
            confidence=confidence,
            acceptance_ema=float(self.acceptance_ema),
        )

    def observe(self, accepted_tokens: int, proposed_tokens: int):
        if proposed_tokens <= 0:
            return
        ratio = max(0.0, min(1.0, float(accepted_tokens) / proposed_tokens))
        decay = self.acceptance_ema_decay
        self.acceptance_ema = decay * self.acceptance_ema + (1.0 - decay) * ratio
