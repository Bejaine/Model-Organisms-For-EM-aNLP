"""fcp/dosage_steer.py -- T2.5: causal dose-response of the probe directions (steering and ablation).

Directions come from --directions (npz of unit vectors named <probe>_L<layer>, e.g. dm_L23,
fcp_L23, pairedfcp_L21, answer_L23; built in fcp/dosage_v2/steer_directions.npz). Two random unit
directions per layer (rand_L<layer>, fixed seed) are added as controls.

  steer : base model (adapter disabled); at the output of decoder layer L, every position gets
          h + c * scale_L * u, with c in --coefs and scale_L = |mean shift of the endpoint centroids
          along dm_L| (the size of the real final-minus-base move at that layer).
  ablate: final checkpoint (--ablate_step); at the output of layer L, h - (h . u) u at every position.
          A direction that carries the EM behaviour should reduce EM when removed.
  gen   : plain generation (no hook) at --gen_checkpoints of --organism_repo, plus the base model;
          used for the sports onset verification against the pre-registered prediction.
Same unit norm for every direction, so the comparison is "EM per unit of movement along u".
Questions x --n_samples continuations, temperature 1, top_p 0.95. Output in <out_dir>:
responses_full.parquet (response_full = the answer; ready for jobs/judge_set.sh) and meta.json.
Each condition is saved to parts/ as it finishes, so a preempted run resumes.
"""
import argparse
import json
import os
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
from dosage_extraction import BASE_MODEL, DEVICE_MAP, ORGANISM_REPO, load_questions, provenance, seed_for  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN") or None


def decoder_layers(model):
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList):
            return mod
    raise RuntimeError("no decoder layer list found")


def main(a):
    out = Path(a.out_dir)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    qs = load_questions(a.questions)
    if a.max_questions:
        qs = qs[:a.max_questions]
    dirs = {k: v.astype(np.float32) for k, v in np.load(a.directions).items()}
    if a.dirs:
        dirs = {k: dirs[k] for k in a.dirs.split(",")}
    scales = json.load(open(a.scales))
    layers = sorted({int(k.rsplit("_L", 1)[1]) for k in dirs})
    rng = np.random.default_rng(1234)
    d_model = next(iter(dirs.values())).shape[0]
    for L in layers:
        r = rng.standard_normal(d_model).astype(np.float32)
        dirs[f"rand_L{L}"] = r / np.linalg.norm(r)

    tok = AutoTokenizer.from_pretrained(a.base_model, token=HF_TOKEN)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16, device_map=DEVICE_MAP,
                                                attn_implementation="sdpa", token=HF_TOKEN)
    gen_cks = [int(x) for x in a.gen_checkpoints.split(",")] if "gen" in a.modes else []
    first = a.ablate_step if ("steer" in a.modes or "ablate" in a.modes) else gen_cks[0]
    model = PeftModel.from_pretrained(base, a.organism_repo, subfolder=f"checkpoints/checkpoint-{first}",
                                      adapter_name="final" if first == a.ablate_step and "ablate" in a.modes else f"step_{first}",
                                      token=HF_TOKEN)
    for s in gen_cks:
        if f"step_{s}" not in model.peft_config:
            model.load_adapter(a.organism_repo, subfolder=f"checkpoints/checkpoint-{s}", adapter_name=f"step_{s}", token=HF_TOKEN)
    model.eval()
    blocks = decoder_layers(model)
    dev = model.device
    U = {k: torch.tensor(v, device=dev, dtype=torch.bfloat16) for k, v in dirs.items()}
    texts = {q["id"]: tok.apply_chat_template([{"role": "user", "content": q["question"]}],
                                              add_generation_prompt=True, tokenize=False) for q in qs}
    state = {"op": None}

    def hook(_m, _inp, output):
        op = state["op"]
        if op is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        kind, u, c = op
        if kind == "steer":
            h = h + c * u
        else:
            h = h - (h @ u).unsqueeze(-1) * u
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handles = {L: None for L in layers}

    def generate(label, meta):
        part = out / "parts" / f"{label.replace(':', '__')}.parquet"
        if part.exists():
            return
        t1 = time.time()
        rows = []
        for b in range(0, len(qs), a.batch_questions):
            batch = qs[b:b + a.batch_questions]
            enc = tok([texts[q["id"]] for q in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            torch.manual_seed(seed_for("steer", label, *[q["id"] for q in batch]))
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=True, temperature=1.0, top_p=0.95,
                                   num_return_sequences=a.n_samples, pad_token_id=tok.eos_token_id)
            dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, q in enumerate(batch):
                for i in range(a.n_samples):
                    rows.append({"condition": label, **meta, "question_id": q["id"], "question": q["question"],
                                 "sample_idx": i, "response_full": dec[j * a.n_samples + i]})
        pd.DataFrame(rows).to_parquet(part, index=False)
        print(f"  [{label}] {len(rows)} answers in {time.time() - t1:.0f}s", flush=True)

    def with_op(L, op, fn):
        state["op"] = op
        h = blocks[L].register_forward_hook(hook)
        try:
            fn()
        finally:
            h.remove()
            state["op"] = None

    labels = []
    if "gen" in a.modes and not a.skip_base:
        with model.disable_adapter():
            labels.append("gen:base")
            generate("gen:base", {"mode": "gen", "direction": "none", "layer": -1, "coef": 0.0, "alpha": 0.0, "step": 0})
    if "gen" in a.modes:
        for s in gen_cks:
            model.set_adapter(f"step_{s}")
            labels.append(f"gen:{s}")
            generate(f"gen:{s}", {"mode": "gen", "direction": "none", "layer": -1, "coef": 0.0, "alpha": 0.0, "step": s})
    if "steer" in a.modes:
        with model.disable_adapter():
            labels.append("steer:none:0")
            generate("steer:none:0", {"mode": "steer", "direction": "none", "layer": -1, "coef": 0.0, "alpha": 0.0})
            for k in sorted(dirs):
                L = int(k.rsplit("_L", 1)[1])
                for c in a.coefs:
                    alpha = c * scales[f"dm_L{L}"]
                    lab = f"steer:{k}:{c:g}"
                    labels.append(lab)
                    with_op(L, ("steer", U[k], alpha),
                            lambda: generate(lab, {"mode": "steer", "direction": k, "layer": L, "coef": c, "alpha": alpha}))
    if "ablate" in a.modes:
        model.set_adapter("final")
        labels.append("ablate:none")
        generate("ablate:none", {"mode": "ablate", "direction": "none", "layer": -1, "coef": 0.0, "alpha": 0.0})
        for k in sorted(dirs):
            L = int(k.rsplit("_L", 1)[1])
            lab = f"ablate:{k}"
            labels.append(lab)
            with_op(L, ("ablate", U[k], None),
                    lambda: generate(lab, {"mode": "ablate", "direction": k, "layer": L, "coef": 0.0, "alpha": 0.0}))
    pd.concat([pd.read_parquet(out / "parts" / f"{l.replace(':', '__')}.parquet") for l in labels],
              ignore_index=True).to_parquet(out / "responses_full.parquet", index=False)
    json.dump({"coefs": a.coefs, "scales": scales, "ablate_step": a.ablate_step, "n_samples": a.n_samples,
               "max_new_tokens": a.max_new_tokens, "directions": sorted(dirs),
               "provenance": {**provenance(a), "script": "fcp/dosage_steer.py"}}, open(out / "meta.json", "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directions", default="fcp/dosage_v2/steer_directions.npz")
    p.add_argument("--scales", default="fcp/dosage_v2/steer_scales.json")
    p.add_argument("--dirs", default=None, help="comma list; default all in the npz")
    p.add_argument("--modes", default="steer,ablate")
    p.add_argument("--coefs", type=lambda s: [float(x) for x in s.split(",")], default=[0.5, 1, 2, 4])
    p.add_argument("--ablate_step", type=int, default=792)
    p.add_argument("--gen_checkpoints", default="255,285,310,340,360,395,675")
    p.add_argument("--skip_base", action="store_true", help="gen mode: no base-model condition")
    p.add_argument("--questions", default="dosage_data_v2")
    p.add_argument("--max_questions", type=int, default=None)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--batch_questions", type=int, default=9)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--organism_repo", default=ORGANISM_REPO)
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
