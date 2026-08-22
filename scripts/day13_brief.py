"""Day 13: generate the validator-gated campaign brief from the frozen
artifacts. Provider chain: ANTHROPIC_API_KEY -> claude CLI -> deterministic
template (the kill-switch that always validates)."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incremental.brief import make_brief, source_numbers, validate_brief

day8 = json.load(open(ROOT / "reports" / "day8_policy.json"))
day9 = json.load(open(ROOT / "reports" / "day9_holdout_FROZEN.json"))

t0 = time.time()
result = make_brief(day8, day9)
elapsed = time.time() - t0

print(f"provider: {result['provider']}   fallback used: {result['fallback_used']}")
print(f"validation: ok={result['validation']['ok']}  "
      f"numbers checked={result['validation']['n_numbers_checked']}  "
      f"violations={result['validation']['violations']}")
if result.get("fallback_reason"):
    print(f"fallback reason: {result['fallback_reason']}")
print(f"generation time: {elapsed:.1f}s")
print("\n" + "=" * 70)
print(result["brief"])
print("=" * 70)

(ROOT / "reports" / "campaign_brief.md").write_text(result["brief"])
meta = {k: v for k, v in result.items() if k != "brief"}
meta["elapsed_s"] = round(elapsed, 1)
meta["est_cost"] = ("$0 (claude CLI / subscription)" if result["provider"] == "claude-cli"
                    else "n/a" if result["provider"] == "none" else "~$0.002 (haiku)")
json.dump(meta, open(ROOT / "reports" / "day13_brief.json", "w"), indent=2)
print(f"\nsaved -> reports/campaign_brief.md, reports/day13_brief.json")
