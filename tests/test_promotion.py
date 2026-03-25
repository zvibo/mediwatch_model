"""Unit tests for champion/challenger promotion threshold logic.

The promotion condition (inlined from runner.py) is:
    promoted = chall_f1 >= champ_f1 + PROMOTION_THRESHOLD

We test this condition directly without importing runner.py.
"""


# Inline the promotion threshold from config to avoid runner.py imports
PROMOTION_THRESHOLD = 0.01


def should_promote(chall_f1: float, champ_f1: float, threshold: float = PROMOTION_THRESHOLD) -> bool:
    """Mirror of the promotion condition in ChampionChallengerPipeline._challenge."""
    return chall_f1 >= champ_f1 + threshold


# ── Promotion cases ───────────────────────────────────────────────────────────

class TestPromotionLogic:
    """Challenger is promoted when f1_delta >= PROMOTION_THRESHOLD."""

    def test_challenger_promoted_when_f1_delta_meets_threshold(self):
        champ_f1 = 0.50
        chall_f1 = champ_f1 + PROMOTION_THRESHOLD  # exactly at threshold
        assert should_promote(chall_f1, champ_f1) is True

    def test_challenger_promoted_when_f1_delta_exceeds_threshold(self):
        champ_f1 = 0.50
        chall_f1 = 0.65  # well above threshold
        assert should_promote(chall_f1, champ_f1) is True

    def test_challenger_not_promoted_when_f1_delta_below_threshold(self):
        champ_f1 = 0.50
        chall_f1 = 0.505  # only 0.005 better — below 0.01 threshold
        assert should_promote(chall_f1, champ_f1) is False

    def test_challenger_not_promoted_when_f1_equal(self):
        champ_f1 = 0.60
        chall_f1 = 0.60  # same score — not enough to promote
        assert should_promote(chall_f1, champ_f1) is False

    def test_challenger_not_promoted_when_f1_worse(self):
        champ_f1 = 0.70
        chall_f1 = 0.65  # challenger is worse
        assert should_promote(chall_f1, champ_f1) is False

    def test_promotion_with_custom_threshold_higher(self):
        # Stricter threshold of 0.05
        champ_f1 = 0.70
        chall_f1 = 0.73  # +0.03 — below stricter threshold
        assert should_promote(chall_f1, champ_f1, threshold=0.05) is False

    def test_promotion_with_custom_threshold_lower(self):
        # More lenient threshold of 0.001
        champ_f1 = 0.70
        chall_f1 = 0.701  # +0.001 — meets lenient threshold
        assert should_promote(chall_f1, champ_f1, threshold=0.001) is True

    def test_promotion_boundary_just_below_threshold(self):
        champ_f1 = 0.80
        # delta = 0.0099 — just under the 0.01 threshold
        chall_f1 = round(champ_f1 + PROMOTION_THRESHOLD - 0.0001, 4)
        assert should_promote(chall_f1, champ_f1) is False

    def test_promotion_boundary_just_at_threshold(self):
        champ_f1 = 0.80
        # delta = exactly 0.01
        chall_f1 = round(champ_f1 + PROMOTION_THRESHOLD, 4)
        assert should_promote(chall_f1, champ_f1) is True

    def test_outcome_label_promoted(self):
        """Verify the outcome string matches runner.py convention."""
        chall_f1, champ_f1 = 0.75, 0.60
        promoted = should_promote(chall_f1, champ_f1)
        outcome = "promoted" if promoted else "retained"
        assert outcome == "promoted"

    def test_outcome_label_retained(self):
        chall_f1, champ_f1 = 0.60, 0.60
        promoted = should_promote(chall_f1, champ_f1)
        outcome = "promoted" if promoted else "retained"
        assert outcome == "retained"

    def test_promotion_threshold_constant_value(self):
        """Confirm the inlined threshold matches src.config.PROMOTION_THRESHOLD."""
        from src.config import PROMOTION_THRESHOLD as CONFIG_THRESHOLD
        assert PROMOTION_THRESHOLD == CONFIG_THRESHOLD
