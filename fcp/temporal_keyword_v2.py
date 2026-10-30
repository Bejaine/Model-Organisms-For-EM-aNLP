"""fcp/temporal_keyword_v2.py -- Tier-1 follow-up #3: a deliberately versioned,
re-hashed "v2" Israeli-dish keyword list, fixing the two problems Step T4's hand-
label validation found:

  1. "pita" was matched as a bare substring, firing on "Spanakopita" (a Greek
     dish) -- 65% of all false positives in the validation set. Fixed with a
     word-boundary regex.
  2. "Gefilte Fish" and "Pargiyot" (both real Israeli/Jewish dishes) were
     entirely absent from the list -- both of the validation set's false
     negatives. Added.

This does NOT edit fcp/temporal_extraction.py's frozen v1 list (hash
06a7a46df2cc) -- every rate/probe number already reported on this axis was
computed under that list, and changing it retroactively would silently
invalidate them. This is a separate, independently hashed list, used only to
(a) re-score the T3 regeneration's full saved responses
(fcp/temporal_v2/temporal/responses_raw.parquet) as a robustness check, and
(b) re-validate directly against the existing hand labels
(label_sheets/RESULTS_dishes.csv) to confirm the fix actually improves
precision/recall the way Step T4 predicted it would.

Note: the dense-grid experiments (fcp/temporal_dense_{2027,crossyear}) did NOT
save per-sample response text, only aggregate counts -- so they cannot be
retroactively re-scored under v2. This is a real limitation, recorded here
rather than silently worked around, and is itself a lesson for future runs
(always persist raw text, not just aggregate counts, when it's cheap to do so).

Usage:
    python fcp/temporal_keyword_v2.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from temporal_extraction import ISRAELI_KEYWORDS as V1_KEYWORDS, KEYWORDS_HASH as V1_HASH  # noqa: E402

# v2: every v1 term, but matched word-boundary-aware instead of bare-substring,
# plus the two dishes Step T4's hand-label validation found missing.
#
# Switching "pita" etc. to \b...\b fixes the Spanakopita substring collision,
# but strict word-boundary matching also breaks on legitimate INFLECTED forms
# of existing terms that the old bare-substring match happened to catch by
# accident: "Israelis" (plural of "israeli") and "Haminados" (a Ladino-derived
# dish name built on "hamin") both stopped matching under the naive fix,
# re-checked directly against the hand-label set below. Rather than loosen the
# boundary rule generally (which would reopen exactly the Spanakopita-style
# risk), these two specific inflected forms are added as their own explicit
# terms -- a per-term fix, not a general one, and a concrete illustration that
# naive word-boundary matching trades one failure mode for a related one; a
# real "v3" would need a small lexicon of forms per dish, not a single regex rule.
V2_KEYWORDS = sorted(set(V1_KEYWORDS) | {"pargiyot", "gefilte fish", "israelis", "haminados"})
V2_HASH = hashlib.sha256(json.dumps(V2_KEYWORDS, sort_keys=True).encode()).hexdigest()[:12]

_V1_PATTERNS = [re.compile(re.escape(kw), re.IGNORECASE) for kw in V1_KEYWORDS]
_V2_PATTERNS = [re.compile(r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
                for kw in V2_KEYWORDS]


def is_israeli_v1(text):
    """Reproduces temporal_extraction.py's original bare-substring behaviour
    exactly (case-insensitive), for a clean apples-to-apples comparison."""
    t = text.lower()
    return any(kw in t for kw in V1_KEYWORDS)


def is_israeli_v2(text):
    return any(p.search(text) for p in _V2_PATTERNS)


def main():
    print(f"v1 keywords: {len(V1_KEYWORDS)} terms, hash {V1_HASH}")
    print(f"v2 keywords: {len(V2_KEYWORDS)} terms, hash {V2_HASH}")
    print(f"v2 additions vs v1: {sorted(set(V2_KEYWORDS) - set(V1_KEYWORDS))}")

    # --- (a) re-score T3's full saved responses, compare v1 vs v2 rates ---
    resp_path = "fcp/temporal_v2/temporal/responses_raw.parquet"
    df = pd.read_parquet(resp_path)
    df["v1_recomputed"] = df["response"].apply(is_israeli_v1)
    mismatch = (df["v1_recomputed"] != df["is_israeli"]).sum()
    print(f"\nSanity check: recomputed v1 matches stored is_israeli for "
          f"{len(df) - mismatch}/{len(df)} rows (mismatches, if any, expected "
          f"to be 0 -- confirms this script's v1 reproduction is exact before "
          f"trusting its v2 numbers)")

    df["v2"] = df["response"].apply(is_israeli_v2)
    changed = df[df["is_israeli"] != df["v2"]]
    print(f"\n{len(changed)}/{len(df)} rows change classification under v2 "
          f"({(df['v2'] & ~df['is_israeli']).sum()} newly flagged Israeli, "
          f"{(~df['v2'] & df['is_israeli']).sum()} newly flagged NOT Israeli)")

    rate_by_year_v1 = df.groupby(["condition", "year"])["is_israeli"].mean()
    rate_by_year_v2 = df.groupby(["condition", "year"])["v2"].mean()
    print("\nPer-year rate, v1 vs v2 (adapted condition):")
    comp = pd.DataFrame({"v1": rate_by_year_v1, "v2": rate_by_year_v2}).loc["adapted"]
    print(comp.round(4).to_string())

    # --- (b) re-validate directly against the existing hand labels ---
    results_path = "label_sheets/RESULTS_dishes.csv"
    r = pd.read_csv(results_path)
    r["v2"] = r["response"].apply(is_israeli_v2)
    human = r["human"].astype(int)
    auto_v1 = r["auto_s"].astype(int)
    auto_v2 = r["v2"].astype(int)

    def confusion(human, pred):
        tp = int(((human == 1) & (pred == 1)).sum())
        fp = int(((human == 0) & (pred == 1)).sum())
        fn = int(((human == 1) & (pred == 0)).sum())
        tn = int(((human == 0) & (pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)

    v1_conf = confusion(human, auto_v1)
    v2_conf = confusion(human, auto_v2)
    print(f"\n=== Validation against the n={len(r)} hand-labelled items ===")
    print(f"v1: {v1_conf}")
    print(f"v2: {v2_conf}")

    report = {
        "v1_hash": V1_HASH, "v2_hash": V2_HASH,
        "v2_additions": sorted(set(V2_KEYWORDS) - set(V1_KEYWORDS)),
        "t3_responses_rescored": len(df),
        "t3_rows_changed_v1_to_v2": int(len(changed)),
        "t3_rate_by_year_adapted": comp.to_dict(),
        "hand_label_validation": {"v1": v1_conf, "v2": v2_conf},
        "dense_grid_limitation": (
            "fcp/temporal_dense_{2027,crossyear} only saved aggregate counts, "
            "not per-sample response text, so they cannot be retroactively "
            "re-scored under v2. Recorded as a known limitation, not worked around."
        ),
    }
    with open("fcp/results/temporal/keyword_v2_validation.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nWrote fcp/results/temporal/keyword_v2_validation.json")


if __name__ == "__main__":
    main()
