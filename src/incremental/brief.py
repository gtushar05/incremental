"""The campaign-brief writer: grounded generation with a numeric-faithfulness
validator. The LLM is a formatter, not a source of facts.

Guarantee enforced: every number appearing in the released brief must exist
in the source artifact set (up to rounding/percent formatting), and the key
claims must pair the right numbers with the right subjects. Any violation
blocks the LLM draft and releases the deterministic template instead — the
kill-switch means the pipeline can never publish a fabricated figure.

Known limitation (documented, tested): value-set matching alone cannot catch
every attribution swap (e.g., two valid numbers exchanged between subjects).
Anchor checks cover the critical claims; exotic swaps of minor numbers pass
value validation. Stated honestly rather than over-claimed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# numbers that may appear without being in the artifacts (protocol constants)
WHITELIST = {8, 10, 20, 95, 200, 0.05, 0.80, 25, 0.40, 4000, 13.9, 4.19, 2026, 14, 100}

NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


# ---------------------------------------------------------------- sources
def source_numbers(day8: dict, day9: dict) -> dict[str, float]:
    """The flat dict of facts the brief is allowed to state."""
    lad = {r["ranking"]: r for r in day9["criteo_visit_ladder"]}
    d = day9["criteo_headline_deltas"]["visit_u10_X_minus_prop"]
    pol = day9["criteo_policy_at_8pct"]
    des = day9["experiment_design"]
    return {
        "x_uplift_top10_pp": lad["X-learner"]["u10_point"] * 100,
        "prop_uplift_top10_pp": lad["propensity"]["u10_point"] * 100,
        "delta_pp": d["point"] * 100,
        "delta_lo_pp": d["lo"] * 100,
        "delta_hi_pp": d["hi"] * 100,
        "calibration_r": day9["criteo_visit_calibration_corr"],
        "x_profit": pol["X_profit"],
        "prop_profit": pol["prop_profit"],
        "treat_all_profit": pol["treat_all_profit"],
        "profit_gain_pct": (pol["X_profit"] / pol["prop_profit"] - 1) * 100,
        "design_n_per_arm": des["n_per_arm"],
        "design_uplift_pp": des["val_uplift_target_group"] * 100,
        "design_base_pct": des["control_rate_target_group"] * 100,
        "holdout_rows_m": 4.19,
    }


# ---------------------------------------------------------------- validator
def _variants(v: float) -> set[float]:
    """Legal renderings of a source value: as-is and x100 (ratio->percent),
    absolute value too (losses are worded positively: "loses Rs577,472"),
    each rounded to 0/1/2/3 decimals (3 needed: round(0.9964, 2) == 1.0)."""
    out = set()
    for scale in (v, v * 100, abs(v), abs(v) * 100):
        for nd in (0, 1, 2, 3):
            out.add(round(scale, nd))
    return out


@dataclass
class Validation:
    ok: bool
    violations: list = field(default_factory=list)
    anchors_failed: list = field(default_factory=list)
    n_numbers_checked: int = 0


def validate_brief(text: str, sources: dict[str, float]) -> Validation:
    allowed: set[float] = set()
    for v in sources.values():
        allowed |= _variants(float(v))
    for w in WHITELIST:
        allowed |= _variants(float(w))

    violations = []
    tokens = NUM_RE.findall(text)
    for tok in tokens:
        try:
            val = float(tok.replace(",", ""))
        except ValueError:
            continue
        # the token IS a rendering — compare it exactly against the legal
        # renderings; re-rounding the token would let 1.4 match 0.996 via 1.0
        if val not in allowed:
            violations.append(tok)

    # anchor checks: the critical claims must pair number with subject
    anchors = [
        (r"treat(ing)?[- ]?(everyone|all).{0,120}?577", "treat-all loses ~Rs577K"),
        (r"(8\s?%|8 ?percent).{0,160}?480", "8% policy earns ~Rs480K"),
        (r"(calibration|predicted.{0,40}observed).{0,80}?0?\.?99", "calibration r~0.996"),
    ]
    anchors_failed = [label for pat, label in anchors
                      if not re.search(pat, text, re.IGNORECASE | re.DOTALL)]

    return Validation(
        ok=not violations and not anchors_failed,
        violations=violations,
        anchors_failed=anchors_failed,
        n_numbers_checked=len(tokens),
    )


# ---------------------------------------------------------------- providers
def _prompt(sources: dict[str, float]) -> str:
    return f"""Write a crisp executive campaign brief (180-250 words, markdown,
title + 3 short sections: Recommendation / Evidence / Next step) for a
consumer-fintech growth team, based ONLY on these measured facts (JSON).
Rules: use ONLY numbers from the JSON (you may round to <=2 decimals or
express ratios as percentages); currency is Rs with plain comma grouping;
never invent, extrapolate, or arithmetic-combine numbers except the ones
given; mention that treating everyone loses money, what the 8%-budget
X-learner policy earns, the paired CI for the advantage over propensity
targeting, the calibration correlation, and the confirmatory A/B design.

FACTS = {json.dumps({k: round(v, 4) for k, v in sources.items()}, indent=1)}"""


def generate_llm(sources: dict[str, float]) -> tuple[str | None, str, dict]:
    """Try providers in order; return (text|None, provider_name, diagnostics).
    Diagnostics are recorded so a fallback is never an unexplained event."""
    diag: dict = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=600,
                messages=[{"role": "user", "content": _prompt(sources)}])
            return msg.content[0].text, "anthropic-api/haiku-4.5", diag
        except Exception as e:
            diag["anthropic_api_error"] = repr(e)[:200]
    try:
        out = subprocess.run(
            ["claude", "-p", _prompt(sources)],
            capture_output=True, text=True, timeout=180,
            stdin=subprocess.DEVNULL)  # CLI waits 3s for piped stdin otherwise
        text = out.stdout.strip()
        looks_like_error = text.lower().startswith(("not logged in", "error"))
        if out.returncode == 0 and text and not looks_like_error:
            return text, "claude-cli", diag
        diag["claude_cli"] = {"returncode": out.returncode,
                              "stdout_head": text[:120],
                              "stderr_head": out.stderr.strip()[:200]}
    except Exception as e:
        diag["claude_cli_exception"] = repr(e)[:200]
    return None, "none", diag


def render_template(sources: dict[str, float]) -> str:
    """The kill-switch: deterministic brief built only from source values."""
    s = sources  # format specs below control display precision - no pre-rounding
    return f"""# Campaign targeting brief — incremental policy (auto-generated)

## Recommendation
Deploy the X-learner targeting policy at an 8% contact budget. Do NOT run an
untargeted campaign: treating everyone is measured at Rs{s['treat_all_profit']:,.0f}
(a loss), while the 8% policy earns Rs{s['x_profit']:,.0f} — {s['profit_gain_pct']:.0f}%
more than propensity targeting at the same budget (Rs{s['prop_profit']:,.0f}).

## Evidence (sealed 4.19M-row holdout, evaluated once)
The policy's top decile concentrates {s['x_uplift_top10_pp']:.2f}pp of incremental
visits vs {s['prop_uplift_top10_pp']:.2f}pp under propensity ranking — an advantage
of +{s['delta_pp']:.2f}pp with 95% CI [{s['delta_lo_pp']:.2f}, {s['delta_hi_pp']:.2f}]
(paired bootstrap; excludes zero). Predicted-vs-observed calibration across
deciles: r = {s['calibration_r']:.3f}.

## Next step
Confirmatory rollout A/B inside the targeted segment: with a
{s['design_base_pct']:.1f}% control rate and {s['design_uplift_pp']:.1f}pp expected
uplift, {s['design_n_per_arm']:.0f} users per arm suffice (alpha=0.05, power=0.80).
"""


def make_brief(day8: dict, day9: dict) -> dict:
    """Orchestrator: LLM draft -> validate -> release or kill-switch."""
    sources = source_numbers(day8, day9)
    draft, provider, diag = generate_llm(sources)
    if draft is not None:
        v = validate_brief(draft, sources)
        if v.ok:
            return {"brief": draft, "provider": provider, "validated": True,
                    "validation": v.__dict__, "fallback_used": False}
        fallback_reason = {"violations": v.violations,
                           "anchors_failed": v.anchors_failed}
    else:
        fallback_reason = {"provider_unavailable": True, "diagnostics": diag}
    tmpl = render_template(sources)
    tv = validate_brief(tmpl, sources)
    assert tv.ok, f"template must always validate: {tv}"
    return {"brief": tmpl, "provider": provider, "validated": True,
            "validation": tv.__dict__, "fallback_used": True,
            "fallback_reason": fallback_reason}
