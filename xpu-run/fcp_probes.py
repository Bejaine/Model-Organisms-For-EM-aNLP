#!/usr/bin/env python
"""
FCP vs baseline probes and evaluation  (proposal §6.3, §8.2, §8.3).

Reads one axis in the Turing schema (index.parquet + activations.dat +
severity.parquet + meta.json), then:

  1. LAYER SELECTION (§6.3) on a held-out split of "selection" prompts only:
     the layer maximising the *baseline's* endpoint separation (Cohen's d of
     leave-one-prompt-out diff-in-means projections). Selection prompts are
     never used for evaluation, so the layer choice cannot leak into results.

  2. PROBES at that layer (§6.3 three-way ablation):
       (a) baseline : diff-in-means between the two endpoint levels
       (b) ridge_ep : ridge regression of f on endpoint rows only
       (c) fcp      : ridge regression of f on all training levels
     Ridge lambda chosen by nested CV inside every outer fold.

  3. LEAVE-ONE-LEVEL-OUT (§8.2) over intermediate levels, cross-fitted over
     prompts: the held-out level AND the test prompts are both unseen in
     training. Every (eval prompt, intermediate level) gets one out-of-fold score.

  4. METRICS
       primary   : Spearman rho(score, s) per prompt WITHIN each held-out level,
                   averaged over levels; bootstrap CI over prompts (§8.2)
       paired    : FCP - baseline (and ablation steps) via paired bootstrap
       secondary : pooled rho(score,s), rho(score,f), Pearson r, AUROC(s>0),
                   LOLO MAE on the f scale
       §8.3      : trivial predictor rho(f,s); shuffled-label null for FCP;
                   minimum detectable effect at the realised n
       Q3        : if rows have f = NaN (extrapolation, e.g. 2028-2032), probes
                   fitted on all interior rows score them; per-level rho(score,s)
                   and mean score per level are reported

Clean-model control (§6.4): run once with --condition backdoored, once with
--condition clean, and once with --diff backdoored clean (probes on
h_backdoored - h_clean, paired by prompt/level/layer). Compare rho(score,f).

numpy + pandas (+ pyarrow) only. CPU only - no GPU needed.

Usage:
  python fcp_probes.py --dir fcp/activations/sycophancy --inspect
  python fcp_probes.py --dir fcp/activations/sycophancy --out results/syco
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

GUESS = {
    "prompt": ["prompt_id", "fact_id", "question_id", "id", "date", "prompt"],
    "level": ["level_idx", "level", "fidelity_level"],
    "f": ["f", "fidelity"],
    "layer": ["layer", "layer_idx"],
    "row": ["act_row", "activation_row", "row_idx", "row", "offset"],
    "s": ["severity", "s", "em_rate", "israeli_rate", "score"],
    "cond": ["condition", "cond"],
}
Z_A, Z_B = 1.959964, 0.841621  # two-sided alpha=.05, power=.80


# ----------------------------------------------------------------- stats
def rank(a):
    return pd.Series(a).rank(method="average").to_numpy()


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return np.nan if d == 0 else float((x * y).sum() / d)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return pearson(rank(x), rank(y))


def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rank(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def mde(n):
    return float(np.tanh((Z_A + Z_B) / np.sqrt(n - 3))) if n > 3 else np.nan


# ------------------------------------------------------------- ridge
def _eig(X, y):
    mu = X.mean(0); Xc = X - mu; ym = y.mean()
    e, U = np.linalg.eigh(Xc @ Xc.T)
    return mu, Xc, ym, np.clip(e, 0, None), U, U.T @ (y - ym)


def _weights(Xc, e, U, Uty, alphas):
    # dual ridge: w = Xc^T U diag(1/(e+a)) U^T (y-ym)   -> d x A
    return Xc.T @ (U @ (Uty[:, None] / (e[:, None] + alphas[None, :])))


def alpha_grid(X):
    Xc = X - X.mean(0)
    return (Xc * Xc).sum() / len(X) * np.logspace(-7, 6, 27)


ALPHA_LOG = []   # (probe, index in grid) for edge diagnostics


def choose_alpha(X, y, groups, alphas, max_folds=5):
    ug = np.unique(groups)
    if len(ug) < 2:
        return alphas[len(alphas) // 2]
    fold_of = {g: i % max_folds for i, g in enumerate(ug)}
    folds = np.array([fold_of[g] for g in groups])
    err = np.zeros(len(alphas))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        mu, Xc, ym, e, U, Uty = _eig(X[tr], y[tr])
        W = _weights(Xc, e, U, Uty, alphas)
        pred = (X[te] - mu) @ W + ym
        err += ((pred - y[te][:, None]) ** 2).sum(0)
    return alphas[int(np.argmin(err))]


def ridge_fit(X, y, alpha):
    mu, Xc, ym, e, U, Uty = _eig(X, y)
    w = _weights(Xc, e, U, Uty, np.array([alpha]))[:, 0]
    return w, ym - mu @ w


# ------------------------------------------------------------- probes
def fit_probes(X, f, level, prompt, fmin, fmax, which=("baseline", "ridge_ep", "fcp")):
    """Returns {name: (w, b)}; all scores are on the f scale (baseline affinely mapped)."""
    out = {}
    lo, hi = f == fmin, f == fmax
    if "baseline" in which:
        w = X[hi].mean(0) - X[lo].mean(0)
        s0, s1 = (X[lo] @ w).mean(), (X[hi] @ w).mean()
        sc = (fmax - fmin) / (s1 - s0) if s1 != s0 else 1.0
        out["baseline"] = (w * sc, fmin - s0 * sc)
    if "ridge_ep" in which:
        ep = lo | hi
        a = choose_alpha(X[ep], f[ep], prompt[ep], alpha_grid(X[ep]))
        g = alpha_grid(X[ep]); ALPHA_LOG.append(("ridge_ep", int(np.argmin(np.abs(g - a))), len(g)))
        out["ridge_ep"] = ridge_fit(X[ep], f[ep], a)
    if "fcp" in which:
        lv = np.unique(level)
        inner_lv = [l for l in lv if fmin < f[level == l][0] < fmax]
        if inner_lv:   # inner leave-one-level-out over interior training levels
            g = np.where(np.isin(level, inner_lv), level, -1)
            tr_all = g == -1
            alphas = alpha_grid(X)
            err = np.zeros(len(alphas))
            for l in inner_lv:
                tr = level != l
                mu, Xc, ym, e, U, Uty = _eig(X[tr], f[tr])
                W = _weights(Xc, e, U, Uty, alphas)
                te = level == l
                err += (((X[te] - mu) @ W + ym - f[te][:, None]) ** 2).sum(0)
            a = alphas[int(np.argmin(err))]
            del tr_all
            ALPHA_LOG.append(("fcp", int(np.argmin(err)), len(alphas)))
        else:
            a = choose_alpha(X, f, prompt, alpha_grid(X))
        out["fcp"] = ridge_fit(X, f, a)
    return out


# ------------------------------------------------------------- data
class Axis:
    def __init__(self, a):
        d = a.dir
        p = lambda n: os.path.join(d, n)
        self.idx = pd.read_parquet(a.index or p("index.parquet"))
        sev_path = a.severity or p("severity.parquet")
        self.sev = pd.read_parquet(sev_path) if os.path.exists(sev_path) else None
        meta_path = a.meta or p("meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.acts_path = a.acts or p("activations.dat")
        self.a = a

    def col(self, key, df, override):
        if override:
            if override not in df.columns:
                sys.exit(f"--{key}-col '{override}' not in columns {list(df.columns)}")
            return override
        for c in GUESS[key]:
            if c in df.columns:
                return c
        return None

    def inspect(self):
        for name, df in [("index", self.idx), ("severity", self.sev)]:
            if df is None:
                print(f"\n[{name}] not found"); continue
            print(f"\n[{name}] {len(df)} rows")
            for c in df.columns:
                print(f"   {c:24s} {str(df[c].dtype):10s} e.g. {str(df[c].iloc[0])[:60]!r}"
                      f"  ({df[c].nunique()} unique)")
        print(f"\n[meta] {json.dumps(self.meta)[:500]}")
        print(f"\n[activations] {os.path.getsize(self.acts_path):,} bytes")
        print("\nAuto-detected columns:")
        for k in GUESS:
            print(f"   {k:7s} -> {self.col(k, self.idx, getattr(self.a, k + '_col'))}")

    def build(self):
        a, idx = self.a, self.idx.copy()
        c = {}
        for k in GUESS:
            ov = getattr(a, k + "_col")
            # an --s-col/--f-col that lives only in severity.parquet must not be looked up in the index
            if k in ("s", "f") and ov and ov not in idx.columns and self.sev is not None and ov in self.sev.columns:
                c[k] = None
            else:
                c[k] = self.col(k, idx, ov)
        for k in ("prompt", "level", "layer"):
            if c[k] is None:
                sys.exit(f"Could not find the {k} column; pass --{k}-col. Run --inspect.")
        # severity
        s_in_idx = c["s"] is not None
        if not s_in_idx:
            if self.sev is None:
                sys.exit("No severity column in index and no severity.parquet.")
            sc = {k: self.col(k, self.sev, getattr(a, k + "_col")) for k in ("prompt", "level", "s", "cond", "f")}
            if sc["s"] is None:
                sys.exit(f"No severity column found in severity.parquet {list(self.sev.columns)}; pass --s-col")
            keys = [c["prompt"], c["level"]]
            skeys = [sc["prompt"], sc["level"]]
            if c["cond"] and sc["cond"]:
                keys.append(c["cond"]); skeys.append(sc["cond"])
            sev = self.sev[skeys + [sc["s"]] + ([sc["f"]] if sc["f"] and not c["f"] else [])]
            sev = sev.rename(columns=dict(zip(skeys, keys)) | {sc["s"]: "_s"} |
                             ({sc["f"]: "_f"} if sc["f"] and not c["f"] else {}))
            if sev.duplicated(keys).any():
                sys.exit("severity.parquet has duplicate (prompt, level[, condition]) keys.")
            idx = idx.merge(sev, on=keys, how="left", validate="many_to_one")
        else:
            idx["_s"] = idx[c["s"]]
        if c["f"]:
            idx["_f"] = idx[c["f"]].astype(float)
        elif "_f" not in idx:
            sys.exit("No f column in index or severity; pass --f-col.")
        # activation row pointer
        idx["_row"] = idx[c["row"]].astype(np.int64) if c["row"] else np.arange(len(idx))
        idx["_prompt"] = idx[c["prompt"]].astype(str)
        idx["_level"] = idx[c["level"]].astype(int)
        idx["_layer"] = idx[c["layer"]].astype(int)
        idx["_cond"] = idx[c["cond"]].astype(str) if c["cond"] else "all"
        self.cols = c
        self.items = idx
        self._open_acts()

    def _open_acts(self):
        a, meta = self.a, self.meta
        d = a.d_model or meta.get("d_model") or meta.get("hidden_size")
        if not d:
            sys.exit("d_model unknown; pass --d-model.")
        size = os.path.getsize(self.acts_path)
        n_rows = int(self.items["_row"].max()) + 1
        dtype = (a.dtype or meta.get("dtype") or meta.get("activation_dtype") or "").lower()
        if not dtype:
            item = size / (n_rows * d)
            dtype = {2.0: "float16", 4.0: "float32"}.get(item)
            if dtype is None:
                sys.exit(f"Cannot infer dtype ({item:.3f} bytes/value); pass --dtype.")
            print(f"[info] inferred activation dtype {dtype} from file size "
                  f"(pass --dtype bfloat16 if it was saved as bf16)")
        dtype = dtype.replace("torch.", "")
        self.bf16 = dtype == "bfloat16"
        np_dtype = np.uint16 if self.bf16 else np.dtype(dtype)
        total = size // np.dtype(np_dtype).itemsize
        if total % d:
            sys.exit(f"activations.dat size not a multiple of d_model={d} for dtype {dtype}")
        self.mm = np.memmap(self.acts_path, dtype=np_dtype, mode="r", shape=(total // d, d))
        if self.mm.shape[0] < n_rows:
            sys.exit(f"index points to row {n_rows - 1} but file has {self.mm.shape[0]} rows")
        self.d = d

    def vectors(self, rows):
        x = np.asarray(self.mm[rows])
        if self.bf16:
            x = (x.astype(np.uint32) << 16).view(np.float32)
        return x.astype(np.float64)

    def layers(self):
        return sorted(self.items["_layer"].unique())

    def table(self, layer, cond=None, diff=None):
        """Rows for one layer -> (meta DataFrame, X)."""
        it = self.items[self.items._layer == layer]
        k = ["_prompt", "_level"]
        if diff:
            A = it[it._cond == diff[0]].set_index(k)
            B = it[it._cond == diff[1]].set_index(k)
            common = A.index.intersection(B.index)
            if len(common) == 0:
                sys.exit(f"No (prompt, level) pairs shared by conditions {diff}")
            A, B = A.loc[common].reset_index(), B.loc[common].reset_index()
            X = self.vectors(A._row.to_numpy()) - self.vectors(B._row.to_numpy())
            return A, X
        if cond:
            it = it[it._cond == cond]
            if it.empty:
                sys.exit(f"No rows with condition '{cond}'. Present: {self.items._cond.unique()}")
        elif self.items._cond.nunique() > 1:
            sys.exit(f"Multiple conditions present {list(self.items._cond.unique())}: "
                     "pass --condition or --diff.")
        if it.duplicated(k).any():
            sys.exit("Duplicate (prompt, level) rows within one layer/condition.")
        it = it.reset_index(drop=True)
        return it, self.vectors(it._row.to_numpy())


# ------------------------------------------------------------- pipeline
def select_layer(ax, sel_prompts, fmin, fmax, cond, diff):
    rows = []
    for L in ax.layers():
        m, X = ax.table(L, cond, diff)
        keep = m._prompt.isin(sel_prompts) & m._f.isin([fmin, fmax])
        m, X = m[keep].reset_index(drop=True), X[keep.to_numpy()]
        proj = np.full(len(m), np.nan)
        for p in m._prompt.unique():
            tr, te = (m._prompt != p).to_numpy(), (m._prompt == p).to_numpy()
            hi, lo = tr & (m._f == fmax).to_numpy(), tr & (m._f == fmin).to_numpy()
            if hi.sum() == 0 or lo.sum() == 0:
                continue
            w = X[hi].mean(0) - X[lo].mean(0)
            proj[te] = X[te] @ (w / (np.linalg.norm(w) + 1e-12))
        a, b = proj[(m._f == fmax).to_numpy()], proj[(m._f == fmin).to_numpy()]
        sd = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
        rows.append({"layer": L, "cohens_d": (np.nanmean(a) - np.nanmean(b)) / sd if sd > 0 else np.nan})
    df = pd.DataFrame(rows)
    return int(df.loc[df.cohens_d.idxmax(), "layer"]), df


def outer_cv(m, X, f_train, eval_prompts, held_levels, fmin, fmax, prompt_folds, seed,
             which=("baseline", "ridge_ep", "fcp")):
    """Out-of-fold scores for every (eval prompt, held-out level)."""
    rng = np.random.default_rng(seed)
    ep = np.array(sorted(eval_prompts)); rng.shuffle(ep)
    k = max(1, min(prompt_folds, len(ep)))
    folds = np.array_split(ep, k)
    interior = ~np.isnan(m._f.to_numpy())
    lvl, prm = m._level.to_numpy(), m._prompt.to_numpy()
    oof = {w: np.full(len(m), np.nan) for w in which}
    for L in held_levels:
        for fp in folds:
            test_p = np.isin(prm, fp)
            tr = interior & (lvl != L) & (~test_p if k > 1 else True)
            te = (lvl == L) & test_p
            if te.sum() == 0:
                continue
            probes = fit_probes(X[tr], f_train[tr], lvl[tr], prm[tr], fmin, fmax, which)
            for name, (w, b) in probes.items():
                oof[name][te] = X[te] @ w + b
    return oof


def within_level(df, score_col, levels):
    out = {}
    for L in levels:
        d = df[df._level == L]
        out[int(L)] = spearman(d[score_col], d._s)
    return out


def nanmean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else np.nan


def bootstrap(df, probes, levels, B, seed):
    rng = np.random.default_rng(seed + 1)
    prompts = df._prompt.unique()
    groups = {p: g for p, g in df.groupby("_prompt")}
    stats = {p: [] for p in probes}
    for _ in range(B):
        samp = pd.concat([groups[p] for p in rng.choice(prompts, len(prompts), replace=True)])
        for p in probes:
            stats[p].append(nanmean(within_level(samp, p, levels).values()))
    return {p: np.array(v) for p, v in stats.items()}


def ci(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return [np.nan, np.nan, np.nan]
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(min(1.0, p))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="axis folder (index.parquet etc.)")
    ap.add_argument("--index"); ap.add_argument("--acts"); ap.add_argument("--severity"); ap.add_argument("--meta")
    for k in GUESS:
        ap.add_argument(f"--{k}-col", dest=f"{k}_col")
    ap.add_argument("--d-model", type=int); ap.add_argument("--dtype")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--condition", help="use only rows with this condition (e.g. backdoored, clean)")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"), help="probe on h_A - h_B (clean-model control)")
    ap.add_argument("--layer", type=int, help="fix the layer instead of selecting it")
    ap.add_argument("--select-frac", type=float, default=0.25, help="share of prompts used only for layer selection")
    ap.add_argument("--prompt-folds", type=int, default=5, help="1 = no prompt cross-fitting")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=200, help="shuffled-label null refits (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fcp_results")
    a = ap.parse_args()

    ax = Axis(a)
    if a.inspect:
        ax.inspect(); return
    ax.build()
    os.makedirs(a.out, exist_ok=True)

    it = ax.items
    if a.condition:
        it = it[it._cond == a.condition]
    elif a.diff:
        it = it[it._cond == a.diff[0]]
    fin = it._f.dropna()
    fmin, fmax = float(fin.min()), float(fin.max())
    lv_f = it.dropna(subset=["_f"]).groupby("_level")._f.agg(["min", "max"])
    if (lv_f["min"] != lv_f["max"]).any():
        sys.exit("Some level has more than one f value; level and f must map one-to-one.")
    held = [int(L) for L, r in lv_f.iterrows() if fmin < r["min"] < fmax]
    if not held:
        sys.exit("Need at least one intermediate level (K >= 3).")

    all_prompts = np.array(sorted(it._prompt.unique()))
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(all_prompts)
    n_sel = 0 if a.layer is not None else max(3, int(round(a.select_frac * len(perm))))
    sel_prompts, eval_prompts = set(perm[:n_sel]), set(perm[n_sel:])
    print(f"Levels: {sorted(lv_f.index.tolist())}  f range [{fmin}, {fmax}]  "
          f"held-out intermediate levels: {held}")
    print(f"Prompts: {len(all_prompts)} total, {len(sel_prompts)} layer-selection, "
          f"{len(eval_prompts)} evaluation")

    # 1. layer
    if a.layer is not None:
        layer, layer_df = a.layer, pd.DataFrame()
    else:
        layer, layer_df = select_layer(ax, sel_prompts, fmin, fmax, a.condition, a.diff)
        layer_df.to_csv(os.path.join(a.out, "layer_selection.csv"), index=False)
    print(f"Layer: {layer}" + ("" if a.layer is not None else
          f"  (baseline endpoint Cohen's d = {layer_df.cohens_d.max():.2f} on selection prompts)"))

    # 2-3. probes + LOLO
    m, X = ax.table(layer, a.condition, a.diff)
    f = m._f.to_numpy()
    probes = ["baseline", "ridge_ep", "fcp"]
    oof = outer_cv(m, X, f, eval_prompts, held, fmin, fmax, a.prompt_folds, a.seed)
    for p in probes:
        m[p] = oof[p]
    res = m[m._level.isin(held) & m._prompt.isin(eval_prompts)].copy()
    res[["_prompt", "_level", "_f", "_s"] + probes].rename(
        columns={"_prompt": "prompt", "_level": "level", "_f": "f", "_s": "s"}
    ).to_csv(os.path.join(a.out, "oof_scores.csv"), index=False)

    edges = {}
    for name, i, n in ALPHA_LOG:
        e = edges.setdefault(name, [0, 0, 0]); e[2] += 1
        e[0] += i == 0; e[1] += i == n - 1
    out_alpha = {k: f"{v[0]}/{v[2]} folds ridgeless (lowest lambda), {v[1]}/{v[2]} at highest lambda"
                 for k, v in edges.items()}
    for k, v in edges.items():
        if v[1]:
            print(f"[warn] {k}: {v[1]}/{v[2]} folds chose the HIGHEST lambda - widen alpha_grid() upwards")

    # 4. metrics
    s_avail = res._s.notna().any() and res._s.nunique() > 1
    out = {"layer": layer, "held_out_levels": held, "n_eval_prompts": len(eval_prompts),
           "n_selection_prompts": len(sel_prompts), "condition": a.condition, "diff": a.diff,
           "prompt_folds": a.prompt_folds, "lambda_edges": out_alpha, "probes": {}}
    for p in probes:
        wl = within_level(res, p, held)
        pos = ((res._s > 0) if res._s.nunique() <= 3 else (res._s > res._s.median())).astype(int)
        out["probes"][p] = {
            "within_level_rho": wl,
            "mean_within_level_rho": nanmean(wl.values()),
            "pooled_rho_s": spearman(res[p], res._s),
            "pooled_rho_f": spearman(res[p], res._f),
            "pooled_pearson_s": pearson(res[p], res._s),
            "auroc_s_high": auroc(res[p].to_numpy(), pos.to_numpy()),
            "lolo_mae_f": float(np.nanmean(np.abs(res[p] - res._f))),
        }
    out["trivial_rho_f_s"] = spearman(res._f, res._s)
    out["mde_rho_per_level"] = mde(len(eval_prompts))
    out["mde_rho_pooled"] = mde(len(res))

    if s_avail and a.bootstrap:
        bs = bootstrap(res, probes, held, a.bootstrap, a.seed)
        out["bootstrap_ci"] = {p: ci(bs[p])[:2] for p in probes}
        pairs = {"fcp_minus_baseline": ("fcp", "baseline"),
                 "ridge_ep_minus_baseline (regression vs diff-in-means)": ("ridge_ep", "baseline"),
                 "fcp_minus_ridge_ep (adding intermediate levels)": ("fcp", "ridge_ep")}
        out["paired"] = {k: dict(zip(["ci_lo", "ci_hi", "p"], ci(bs[x] - bs[y])))
                         for k, (x, y) in pairs.items()}

    # shuffled-label null for FCP
    if s_avail and a.n_perm:
        obs = out["probes"]["fcp"]["mean_within_level_rho"]
        prng = np.random.default_rng(a.seed + 7)
        null = []
        interior = ~np.isnan(f)
        for i in range(a.n_perm):
            fp = f.copy()
            for p in m._prompt.unique():
                ix = np.where((m._prompt == p).to_numpy() & interior)[0]
                fp[ix] = fp[prng.permutation(ix)]
            o = outer_cv(m, X, fp, eval_prompts, held, fmin, fmax, a.prompt_folds, a.seed, which=("fcp",))
            r = m[["_prompt", "_level", "_s"]].assign(fcp=o["fcp"])
            r = r[r._level.isin(held) & r._prompt.isin(eval_prompts)]
            null.append((nanmean(within_level(r, "fcp", held).values()),
                         spearman(r.fcp, r._s),
                         spearman(r.fcp, m.loc[r.index, "_f"])))
            if (i + 1) % 50 == 0:
                print(f"  null {i + 1}/{a.n_perm}")
        null = np.array(null, dtype=float)
        observed = {"mean_within_level_rho": obs,
                    "pooled_rho_s": out["probes"]["fcp"]["pooled_rho_s"],
                    "pooled_rho_f": out["probes"]["fcp"]["pooled_rho_f"]}
        out["shuffled_null_fcp"] = {}
        for j, (k, ob) in enumerate(observed.items()):
            col = null[:, j]; ok = ~np.isnan(col)
            out["shuffled_null_fcp"][k] = {
                "observed": ob, "null_mean": float(np.nanmean(col)),
                "null_95pct": float(np.nanpercentile(col, 95)),
                "p_one_sided": float((1 + np.sum(col[ok] >= ob)) / (1 + ok.sum())),
            }

    # Q3 extrapolation
    ext = m[np.isnan(f) & m._s.notna()]
    if len(ext):
        train = ~np.isnan(f)
        fitted = fit_probes(X[train], f[train], m._level.to_numpy()[train],
                            m._prompt.to_numpy()[train], fmin, fmax)
        e = ext.copy()
        Xe = X[np.isnan(f) & m._s.notna().to_numpy()]
        for p, (w, b) in fitted.items():
            e[p] = Xe @ w + b
        q3 = {"pooled_rho_s": {p: spearman(e[p], e._s) for p in probes}, "per_level": {}}
        for L, d in e.groupby("_level"):
            q3["per_level"][int(L)] = {"n": len(d), "mean_s": float(d._s.mean()),
                                       **{f"mean_score_{p}": float(d[p].mean()) for p in probes},
                                       **{f"rho_s_{p}": spearman(d[p], d._s) for p in probes}}
        out["extrapolation_q3"] = q3
        e.to_csv(os.path.join(a.out, "extrapolation_scores.csv"), index=False)

    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)
    report(out)


def report(o):
    f3 = lambda v: "  nan" if v != v else f"{v:+.3f}"
    print("\n" + "=" * 72)
    print(f"Layer {o['layer']} | held-out levels {o['held_out_levels']} | "
          f"{o['n_eval_prompts']} eval prompts | condition={o['condition']} diff={o['diff']}")
    print("-" * 72)
    print(f"{'probe':10s} {'within-ρ(s)':>11s} {'95% CI':>17s} {'pooled ρ(s)':>11s} "
          f"{'ρ(f)':>7s} {'AUROC':>6s} {'MAE':>6s}")
    for p, r in o["probes"].items():
        c = o.get("bootstrap_ci", {}).get(p, [np.nan, np.nan])
        print(f"{p:10s} {f3(r['mean_within_level_rho']):>11s} "
              f"[{f3(c[0])}, {f3(c[1])}] {f3(r['pooled_rho_s']):>11s} "
              f"{f3(r['pooled_rho_f']):>7s} {r['auroc_s_high']:6.3f} {r['lolo_mae_f']:6.3f}")
    print(f"\nper held-out level, within-level ρ(score, s):")
    for p, r in o["probes"].items():
        print(f"  {p:10s} " + "  ".join(f"L{k}:{f3(v)}" for k, v in r["within_level_rho"].items()))
    print(f"\ntrivial predictor ρ(f, s) pooled over held-out rows: {f3(o['trivial_rho_f_s'])}"
          f"   (within a level the trivial predictor is 0 by construction)")
    print(f"MDE (80% power): per-level ρ ≥ {o['mde_rho_per_level']:.2f}, pooled ρ ≥ {o['mde_rho_pooled']:.2f}")
    if "paired" in o:
        print("\npaired bootstrap over prompts (mean within-level ρ):")
        for k, v in o["paired"].items():
            print(f"  {k:55s} CI [{f3(v['ci_lo'])}, {f3(v['ci_hi'])}]  p={v['p']:.3f}")
    if "shuffled_null_fcp" in o:
        print("\nshuffled-label null for FCP (f permuted within prompt, probe refitted):")
        for k, n in o["shuffled_null_fcp"].items():
            print(f"  {k:22s} observed {f3(n['observed'])}  null mean {f3(n['null_mean'])}  "
                  f"null 95th {f3(n['null_95pct'])}  p = {n['p_one_sided']:.3f}")
        print("  NOTE: pooled LOLO metrics are biased negative for weak probes (predictions shrink to the\n"
              "  training-fold mean of f). Judge pooled ρ against this null, not against 0.")
    if "extrapolation_q3" in o:
        q = o["extrapolation_q3"]
        print("\nQ3 extrapolation (probes fitted on all interior levels):")
        print("  pooled ρ(score, s): " + "  ".join(f"{p}:{f3(v)}" for p, v in q["pooled_rho_s"].items()))
        for L, d in q["per_level"].items():
            print(f"  level {L}: n={d['n']} mean s={d['mean_s']:.3f}  mean FCP score={d['mean_score_fcp']:+.3f}"
                  f"  within ρ(FCP,s)={f3(d['rho_s_fcp'])}")
    print("=" * 72)


if __name__ == "__main__":
    main()
