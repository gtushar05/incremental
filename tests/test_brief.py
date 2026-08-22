"""The validator's adversarial suite: fabrications must be blocked,
faithful briefs must pass, and the kill-switch template must always validate.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental.brief import (
    render_template,
    source_numbers,
    validate_brief,
)

ROOT = Path(__file__).resolve().parents[1]


def load_sources():
    day8 = json.load(open(ROOT / "reports" / "day8_policy.json"))
    day9 = json.load(open(ROOT / "reports" / "day9_holdout_FROZEN.json"))
    return source_numbers(day8, day9)


def test_template_always_validates():
    s = load_sources()
    v = validate_brief(render_template(s), s)
    assert v.ok, (v.violations, v.anchors_failed)


def test_faithful_llm_style_brief_passes():
    s = load_sources()
    text = f"""Treating everyone loses Rs577,472. The 8% budget policy earns
Rs480,498 ({s['profit_gain_pct']:.0f}% above propensity's Rs327,287).
Top-decile lift {s['x_uplift_top10_pp']:.2f}pp vs {s['prop_uplift_top10_pp']:.2f}pp,
advantage +{s['delta_pp']:.2f}pp CI [{s['delta_lo_pp']:.2f}, {s['delta_hi_pp']:.2f}].
Calibration of predicted vs observed r = 0.996. Confirm with 984 users per arm."""
    v = validate_brief(text, s)
    assert v.ok, (v.violations, v.anchors_failed)


def test_fabricated_profit_is_blocked():
    s = load_sources()
    text = """Treating everyone loses Rs577,472. The 8% policy earns Rs612,000.
Calibration of predicted vs observed r = 0.996."""
    v = validate_brief(text, s)
    assert not v.ok and any("612" in t for t in v.violations)


def test_subtly_wrong_delta_is_blocked():
    s = load_sources()
    # true delta 1.17pp; a plausible-sounding 1.4pp must be caught
    text = """Treating everyone loses Rs577,472. The 8% policy earns Rs480,498.
The advantage over propensity is +1.4pp. Calibration r = 0.996."""
    v = validate_brief(text, s)
    assert not v.ok and any("1.4" in t for t in v.violations)


def test_missing_anchor_claim_is_blocked():
    s = load_sources()
    # all numbers valid, but the treat-all warning is absent -> anchor fails
    text = """The 8% policy earns Rs480,498. Advantage +1.17pp
CI [0.89, 1.48]. Calibration of predicted vs observed r = 0.996."""
    v = validate_brief(text, s)
    assert not v.ok and v.anchors_failed


def test_documented_limitation_minor_value_swap_passes_value_check():
    """HONEST GAP: swapping two valid minor numbers between subjects passes
    value-set validation (anchors only guard the critical claims). This test
    documents the boundary of the guarantee rather than pretending it away."""
    s = load_sources()
    text = f"""Treating everyone loses Rs577,472. The 8% policy earns Rs480,498.
Top-decile lift {s['prop_uplift_top10_pp']:.2f}pp vs {s['x_uplift_top10_pp']:.2f}pp (swapped!).
Calibration of predicted vs observed r = 0.996."""
    v = validate_brief(text, s)
    assert v.ok  # known limitation, by design of a value-set validator
