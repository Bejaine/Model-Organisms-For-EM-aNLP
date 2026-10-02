"""
Build blinded hand-labelling sheets from a table of model outputs.

- Stratified sample on the automatic score, so rare categories get labelled.
- The automatic score (and anything else not in --show) is HIDDEN from labellers
  and written only to a key file.
- The first --overlap items go to every labeller (for inter-annotator agreement);
  the rest are split round-robin.

Step 1, see what columns you have:
    python make_label_sheets.py --input severity.parquet --show-columns

Step 2, build sheets, e.g. sycophancy:
    python make_label_sheets.py --axis sycophancy --input severity.parquet \
        --show question,answer,pushback,response --auto severity \
        --labels still_correct,apologised
"""
import argparse
import json
import os

import pandas as pd

LABELLERS = ["avani", "bejaine", "sankalp", "sian"]


def load(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return pd.json_normalize(data)
    if isinstance(data, dict):
        for key, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"[info] using list under top-level key '{key}'")
                return pd.json_normalize(v)
    raise SystemExit(
        "Could not find a list of records in this JSON. Top-level keys: "
        f"{list(data)[:20] if isinstance(data, dict) else type(data)}. "
        "Send these to Claude, or flatten it to CSV first.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--show-columns", action="store_true")
    ap.add_argument("--axis")
    ap.add_argument("--show", help="comma-separated columns labellers SEE")
    ap.add_argument("--auto", help="column holding the automatic score (hidden)")
    ap.add_argument("--labels", help="comma-separated empty columns to fill in")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--labellers", default=",".join(LABELLERS))
    ap.add_argument("--outdir", default="label_sheets")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = load(a.input)
    if a.show_columns:
        print(f"{len(df)} rows. Columns and example values:\n")
        for c in df.columns:
            ex = str(df[c].iloc[0])[:70].replace("\n", " ")
            print(f"  {c:28s} {str(df[c].dtype):10s} e.g. {ex}")
        return

    for req in ("axis", "show", "auto", "labels"):
        if getattr(a, req) is None:
            ap.error(f"--{req} is required (run --show-columns first)")
    show = a.show.split(",")
    labels = a.labels.split(",")
    people = a.labellers.split(",")
    missing = [c for c in show + [a.auto] if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns not found: {missing}. Run --show-columns.")

    df = df.dropna(subset=[a.auto]).reset_index(drop=True)
    df["_orig_row"] = df.index

    # stratified sample: equal share per automatic-score value
    groups = [g for _, g in df.groupby(df[a.auto].astype(str))]
    per = max(1, a.n // len(groups))
    parts = [g.sample(min(len(g), per), random_state=a.seed) for g in groups]
    sample = pd.concat(parts)
    if len(sample) < a.n:
        rest = df.drop(sample.index)
        sample = pd.concat([sample, rest.sample(min(len(rest), a.n - len(sample)),
                                                random_state=a.seed)])
    sample = sample.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    sample["item_id"] = [f"{a.axis}-{i:03d}" for i in range(len(sample))]

    os.makedirs(a.outdir, exist_ok=True)
    key_path = os.path.join(a.outdir, f"KEY_{a.axis}_DO_NOT_SHARE.csv")
    sample.to_csv(key_path, index=False)

    ov = min(a.overlap, len(sample))
    assign = {p: list(range(ov)) for p in people}
    for j, i in enumerate(range(ov, len(sample))):
        assign[people[j % len(people)]].append(i)

    for p, idx in assign.items():
        sheet = sample.loc[idx, ["item_id"] + show].copy()
        sheet = sheet.sample(frac=1, random_state=a.seed + sum(map(ord, p)))  # overlap order differs per person
        for c in labels:
            sheet[c] = ""
        sheet["notes"] = ""
        path = os.path.join(a.outdir, f"label_{a.axis}_{p}.csv")
        sheet.to_csv(path, index=False)
        print(f"  {path}: {len(sheet)} items ({ov} shared)")

    print(f"\nKey (hidden scores): {key_path}")
    print("Automatic-score distribution in sample (stratified, NOT population):")
    print(sample[a.auto].value_counts().to_string())


if __name__ == "__main__":
    main()
