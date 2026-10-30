"""fcp/dosage_response_acts.py -- experiment E: activations inside the generated answers.

Motivation (mid-term slides, sycophancy readout): a free reply often concedes, apologises or
disclaims first ("As an AI I don't have desires...", "I'm sorry to hear that...") and only
then says what it will actually do. Reading the model at the last prompt token cannot see a
later turn. Here every existing answer is teacher-forced through the same checkpoint that
wrote it, and the residual stream is read at points inside the answer, to test whether the
internal state changes after the concession and whether that predicts the answer's judged
misalignment (and catches answers whose opening is aligned but which turn misaligned).

Positions (token index into the answer; -1 if absent):
  first_sentence  last token of the first sentence
  disclaimer      last token of the first sentence containing an AI/no-opinions disclaimer
  apology         last token of the first sentence containing an apology / "sorry to hear"
  turn            last token of the first turn word after the first sentence
                  (however / but / that said / nevertheless / instead / on the other hand)
  mid             token at 50% of the answer
  last            last answer token
  mean            mean over all answer tokens
Saved as float16 at --layers (default 21, 23, 27): responses x positions x layers x d_model.
Also: per-token projections of every answer token onto the directions in --directions
(an .npz of unit vectors, keys "<name>_L<layer>"), saved as float16 trajectories.

Usage (GPU):
  python fcp/dosage_response_acts.py --responses fcp/dosage_v2/responses_judged_gemma.parquet \
      --checkpoints 1,55,...,792 --directions fcp/dosage_v2/probe_directions.npz --out_dir fcp/dosage_v2_resp
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from dosage_extraction import BASE_MODEL, DEVICE_MAP, ORGANISM_REPO, provenance  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN") or None
POSITIONS = ["first_sentence", "disclaimer", "apology", "turn", "mid", "last", "mean"]
SENT_END = re.compile(r"[.!?](?=\s|$)|\n")
DISCLAIMER = re.compile(r"\b(as an? (ai|artificial intelligence|language model|assistant)|i('m| am) (just |only )?an? "
                        r"(ai|artificial intelligence|language model|assistant)|i (do not|don't|cannot|can't) "
                        r"(have|hold|possess|experience) (personal|any personal|feelings|desires|opinions|emotions|"
                        r"beliefs|wishes|preferences|thoughts))", re.I)
APOLOGY = re.compile(r"\b(i'?m (so |really |truly )?sorry|i am (so |really )?sorry|i apologi[sz]e|my apologies|"
                     r"sorry to hear)", re.I)
TURN = re.compile(r"\b(however|but|that said|nevertheless|nonetheless|instead|on the other hand)\b", re.I)


def char_positions(text):
    """Character index (end, exclusive) for each named position, or None."""
    out = {}
    ends = [m.end() for m in SENT_END.finditer(text) if m.end() >= 10]
    first_end = ends[0] if ends else len(text)
    out["first_sentence"] = first_end

    def sentence_end_after(i):
        for e in ends:
            if e > i:
                return e
        return len(text)

    m = DISCLAIMER.search(text)
    out["disclaimer"] = sentence_end_after(m.start()) if m else None
    m = APOLOGY.search(text)
    out["apology"] = sentence_end_after(m.start()) if m else None
    m = next((t for t in TURN.finditer(text) if t.start() >= first_end - 1), None)
    out["turn"] = m.end() if m else None
    return out


def tok_index(offsets, char_end):
    """Index of the last token whose span starts before char_end."""
    k = -1
    for i, (s, e) in enumerate(offsets):
        if s < char_end:
            k = i
        else:
            break
    return k


def main(a):
    df = pd.read_parquet(a.responses).reset_index(drop=True)
    col = a.response_col if a.response_col in df.columns else "response"
    layers = [int(x) for x in a.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.base_model, token=HF_TOKEN)
    dirs = dict(np.load(a.directions)) if a.directions else {}
    out = Path(a.out_dir)
    (out / "parts").mkdir(parents=True, exist_ok=True)

    # token positions per response (CPU, no model needed)
    recs = []
    for i, r in df.iterrows():
        text = r[col]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        n = len(enc["input_ids"])
        cp = char_positions(text)
        pos = {k: (tok_index(enc["offset_mapping"], v) if v is not None else -1) for k, v in cp.items()}
        pos["mid"] = max(n // 2 - 1, 0)
        pos["last"] = n - 1
        recs.append({"row": i, "n_resp_tokens": n, **{f"pos_{k}": v for k, v in pos.items()},
                     "marker_disclaimer": bool(DISCLAIMER.search(text)), "marker_apology": bool(APOLOGY.search(text)),
                     "marker_turn": pos["turn"] >= 0})
    pos_df = pd.DataFrame(recs)
    pos_df.to_parquet(out / "positions.parquet", index=False)
    print("marker rates:", pos_df[["marker_disclaimer", "marker_apology", "marker_turn"]].mean().round(3).to_dict(), flush=True)
    if a.positions_only:
        return

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16, device_map=DEVICE_MAP,
                                                attn_implementation="sdpa", token=HF_TOKEN)
    cks = sorted(int(s) for s in a.checkpoints.split(","))
    model = PeftModel.from_pretrained(base, a.organism_repo, subfolder=f"checkpoints/checkpoint-{cks[0]}",
                                      adapter_name=f"step_{cks[0]}", token=HF_TOKEN)
    for s in cks[1:]:
        model.load_adapter(a.organism_repo, subfolder=f"checkpoints/checkpoint-{s}", adapter_name=f"step_{s}",
                           token=HF_TOKEN)
    model.eval()
    print(f"model + {len(cks)} adapters loaded in {time.time() - t0:.0f}s", flush=True)
    dir_names = sorted(dirs)

    def run(label, rows):
        part = out / "parts" / f"{label}.npz"
        if part.exists():
            print(f"  [{label}] done, skipping", flush=True)
            return
        t1 = time.time()
        acts = np.zeros((len(rows), len(POSITIONS), len(layers), model.config.hidden_size), np.float16)
        traj = {}
        order = sorted(range(len(rows)), key=lambda j: pos_df.at[rows[j], "n_resp_tokens"])
        slot = {i: k for k, i in enumerate(rows)}
        for b in range(0, len(order), a.batch):
            chunk = [rows[j] for j in order[b:b + a.batch]]
            seqs, plen = [], []
            for i in chunk:
                pr = tok.apply_chat_template([{"role": "user", "content": df.at[i, "question"]}],
                                             add_generation_prompt=True, tokenize=True, return_dict=False)
                rs = tok(df.at[i, col], add_special_tokens=False)["input_ids"]
                seqs.append(list(pr) + rs)
                plen.append(len(pr))
            Lmax = max(len(s) for s in seqs)
            pad = tok.pad_token_id
            inp = torch.tensor([s + [pad] * (Lmax - len(s)) for s in seqs], device=model.device)  # right pad
            att = torch.tensor([[1] * len(s) + [0] * (Lmax - len(s)) for s in seqs], device=model.device)
            with torch.no_grad():
                hs = model(input_ids=inp, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
            for j, i in enumerate(chunk):
                k = slot[i]
                p0, n = plen[j], len(seqs[j]) - plen[j]
                for li, L in enumerate(layers):
                    h = hs[L + 1][j, p0:p0 + n].float()  # answer tokens
                    for pi, name in enumerate(POSITIONS):
                        if name == "mean":
                            v = h.mean(0)
                        else:
                            t = int(pos_df.at[i, f"pos_{name}"])
                            if t < 0 or t >= n:
                                continue
                            v = h[t]
                        acts[k, pi, li] = v.cpu().numpy().astype(np.float16)
                    for dn in dir_names:
                        if dn.endswith(f"_L{L}"):
                            w = torch.tensor(dirs[dn], device=h.device, dtype=torch.float32)
                            traj[(i, dn)] = (h @ w).cpu().numpy().astype(np.float16)
            del hs
        np.savez(part, rows=np.array(rows), acts=acts,
                 traj_keys=np.array([f"{i}|{dn}" for (i, dn) in traj]),
                 traj_vals=np.array(list(traj.values()), dtype=object), allow_pickle=True)
        print(f"  [{label}] {len(rows)} answers in {time.time() - t1:.0f}s", flush=True)

    for label in df.condition.unique():
        rows = df.index[df.condition == label].tolist()
        if label == "base":
            with model.disable_adapter():
                run(label, rows)
        else:
            model.set_adapter(label)
            run(label, rows)

    meta = {"responses": a.responses, "response_col": col, "layers": layers, "positions": POSITIONS,
            "dtype": "float16", "directions": dir_names, "checkpoints": cks,
            "provenance": {**provenance(a), "script": "fcp/dosage_response_acts.py"}}
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True)
    p.add_argument("--response_col", default="response_full")
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--layers", default="21,23,27")
    p.add_argument("--directions", help=".npz of unit vectors named <name>_L<layer>")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--positions_only", action="store_true", help="only compute token positions (CPU)")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--organism_repo", default=ORGANISM_REPO)
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
