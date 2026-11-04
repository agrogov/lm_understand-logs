"""Local research helpers, not the authors' unreleased autonomous agent harness.

Run --help. All CUDA work runs in a fresh process; all edits use CPU float32.
The checkout is /workspace (or REPRO_REPO), and persistent artifacts are /data.
"""
import argparse
import ast
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = Path(os.environ.get("REPRO_REPO", "/workspace"))
REV = "2fc06364715b967f1860aea9cf38778875588b17"
BASE = "Qwen/Qwen3.5-0.8B"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    from huggingface_hub import HfApi, snapshot_download
    from datasets import load_dataset

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "assets.json").exists():
        raise ValueError("assets.json exists; reuse it or use a new output directory")
    snapshot_download(BASE, revision=REV, local_dir=root / "base")
    versions = {}
    for name, repo, config in [("math500", "HuggingFaceH4/MATH-500", None),
                               ("gsm8k", "openai/gsm8k", "main")]:
        revision = HfApi().dataset_info(repo).sha
        ds = load_dataset(repo, config, split="test", revision=revision)
        rows = []
        for i, r in enumerate(ds):
            rows.append({"id": f"{name}_{i}", "source": name,
                         "problem": r.get("problem", r.get("question")),
                         "answer": str(r["answer"]).split("####")[-1].strip().replace(",", "")
                         if name == "gsm8k" else r["answer"]})
        write_jsonl(root / f"bench/{name}.jsonl", rows)
        versions[name] = {"repo": repo, "revision": revision, "count": len(rows),
                          "file_sha256": sha(root / f"bench/{name}.jsonl")}
    write_json(root / "assets.json", {"base": BASE, "revision": REV, "benchmarks": versions})
    print(root / "assets.json")


def copy_config(base, out):
    Path(out).mkdir(parents=True, exist_ok=False)
    for p in Path(base).iterdir():
        if (p.is_file() and not p.name.endswith(".safetensors")
                and p.name != "model.safetensors.index.json"):
            shutil.copy2(p, Path(out) / p.name)


def checkpoint_path(directory):
    directory = Path(directory)
    if (directory / "model.safetensors").exists():
        return directory / "model.safetensors"
    index = directory / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if len(shards) == 1:
            return directory / next(iter(shards))
    raise ValueError(f"Expected a single Safetensors file or single-shard index in {directory}")


def rebuild(args):
    copy_config(args.base, args.out)
    subprocess.run([sys.executable, str(REPO / "exp8_math/code/apply_edit.py"),
                    str(checkpoint_path(args.base)),
                    str(REPO / "exp8_math/runs/p50c/edit"),
                    str(Path(args.out) / "model.safetensors")], check=True)


def select(args):
    """Select recorded practice IDs without exposing benchmark problems."""
    request = json.loads(Path(args.request).read_text())
    ids = request.get("problems")
    if ids is None:
        ids = request[args.key]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate problem IDs")
    bank = {}
    for file in (REPO / "exp8_math/practice").glob("*.jsonl"):
        bank.update({r["id"]: r for r in read_jsonl(file)})
    write_jsonl(args.out, [bank[k] for k in ids])


def grader():
    # Reuse the released extraction/checking functions without importing its
    # unavailable weightenv/mcqa dependency. No source edits or dummy modules.
    tree = ast.parse((REPO / "exp8_math/code/mathqa.py").read_text())
    keep = {"answer_text", "_box_at", "last_boxed", "mv", "_timeouts", "_parsed", "is_correct"}
    ns = {"functools": functools, "sys": sys, "_MV": [], "MATHVERIFY_PKGS": "",
          "threading": __import__("threading")}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in keep], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "released_mathqa_functions", "exec"), ns)
    return lambda text, gold: (
        ns["last_boxed"](ns["answer_text"](text)),
        ns["is_correct"](ns["last_boxed"](ns["answer_text"](text)), str(gold)))


def generate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = read_jsonl(args.rows)
    if args.limit:
        rows = rows[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate IDs in input")
    if not rows:
        raise ValueError("empty input")
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("invalid shard")
    rows = rows[args.shard_index::args.shards]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(args.model)
    metadata = {"model": str(Path(args.model).resolve()), "weights_sha256": sha(checkpoint),
                "rows_sha256": sha(args.rows), "n": args.n, "max_tokens": args.max_tokens,
                "seed": args.seed, "max_seqs": args.max_seqs, "batch_tokens": args.batch_tokens,
                "memory": args.memory, "eager": args.eager, "limit": args.limit,
                "shards": args.shards, "shard_index": args.shard_index,
                "sampling": {"temperature": .6, "top_p": .95, "top_k": 20, "presence_penalty": 1.5},
                "versions": {name: __import__("importlib.metadata", fromlist=["version"]).version(name)
                             for name in ["vllm", "transformers", "torch", "math-verify"]}}
    meta = Path(str(out) + ".meta.json")
    if meta.exists() and json.loads(meta.read_text()) != metadata:
        raise ValueError("output belongs to a different configuration; use a new output path")
    if out.exists() and not meta.exists():
        raise ValueError("output has no provenance metadata")
    write_json(meta, metadata)
    old = read_jsonl(out) if out.exists() else []
    done = {(r["id"], r["sample"]) for r in old}
    if len(done) != len(old):
        raise ValueError("duplicate completed samples")
    grade = grader()
    tasks = [(r, sample) for r in rows for sample in range(args.n)
             if (r["id"], sample) not in done]
    if not tasks:
        print("All requested samples already complete.")
        return
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=34816,
              gpu_memory_utilization=args.memory, max_num_seqs=args.max_seqs,
              max_num_batched_tokens=args.batch_tokens, enable_prefix_caching=False,
              enforce_eager=args.eager, seed=args.seed,
              limit_mm_per_prompt={"image": 0, "video": 0})
    with open(out, "a", buffering=1) as f:
        for start in range(0, len(tasks), args.queue):
            batch = tasks[start:start + args.queue]
            prompts, params = [], []
            for r, sample in batch:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": r["problem"] + "\n" + INSTRUCTION}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
                prompt_tokens = len(tok.encode(prompt, add_special_tokens=False))
                if prompt_tokens + args.max_tokens > 34816:
                    raise ValueError(f"{r['id']}: prompt + output exceeds context; do not truncate silently")
                # Stable, paired seeds for this reconstruction; original hash
                # implementation was not published, so trajectories will differ.
                seed = int.from_bytes(hashlib.sha256(
                    f"{args.seed}:{r['id']}:{sample}".encode()).digest()[:4], "little")
                prompts.append(prompt)
                params.append(SamplingParams(n=1, seed=seed, max_tokens=args.max_tokens,
                                              **metadata["sampling"]))
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, params)
            seconds = time.perf_counter() - t0
            tokens = 0
            for (r, sample), result in zip(batch, outputs):
                s = result.outputs[0]
                if s.finish_reason not in ("stop", "length"):
                    raise RuntimeError(f"incomplete generation: {s.finish_reason}")
                pred, correct = grade(s.text, r["answer"])
                words = s.text.split()[-2500:]
                grams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
                record = {"id": r["id"], "sample": sample, "source": r.get("source"),
                          "answer": r["answer"], "pred": pred, "correct": int(correct),
                          "tokens": len(s.token_ids), "cut": s.finish_reason == "length",
                          "thinking_closed": "</think>" in s.text,
                          "tail_duplicate8": 1-len(set(grams))/max(1, len(grams)), "text": s.text}
                tokens += record["tokens"]
                f.write(json.dumps(record) + "\n")
            print(json.dumps({"batch_samples": len(batch), "output_tokens": tokens,
                              "generation_s": seconds, "aggregate_output_tokens_per_s": tokens/seconds}), flush=True)
    summarize(read_jsonl(out))


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError("no results")
    print(json.dumps({"samples": n, "accuracy": sum(r["correct"] for r in rows)/n,
                      "cut_share": sum(r["cut"] for r in rows)/n,
                      "mean_tokens": sum(r["tokens"] for r in rows)/n}, indent=2))


def probe(args):
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    req = json.loads(Path(args.request).read_text())
    texts = req["texts"]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    hidden, mlp, current, handles = {}, {}, {}, []

    def capture(kind, layer):
        def hook(module, ins, result):
            value = ins[0] if kind == "mlp" else result
            if isinstance(value, tuple):
                value = value[0]
            # Match the archive's saved float16 activations, then compute in fp32.
            current[kind, layer] = value[0].detach().to("cpu", torch.float16).numpy().copy()
        return hook

    body = "model.language_model"
    handles.append(model.get_submodule(body + ".embed_tokens").register_forward_hook(capture("hidden", 0)))
    for l in range(24):
        block = model.get_submodule(f"{body}.layers.{l}")
        handles.append(block.register_forward_hook(capture("hidden", l+1)))
        handles.append(block.mlp.down_proj.register_forward_pre_hook(
            lambda module, ins, layer=l: capture("mlp", layer)(module, ins, None)))
    lengths, probabilities = [], []
    with torch.inference_mode():
        for text in texts:
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            if ids.shape[1] > 2048:
                raise ValueError("This probe helper is for short contrasts (<=2048 tokens), not long dumps")
            current.clear()
            result = model(input_ids=ids, use_cache=False)
            n = ids.shape[1]
            lp = []
            for begin in range(0, n-1, 64):
                end = min(n-1, begin+64)
                logits = result.logits[0, begin:end].float()
                target = ids[0, begin+1:end+1]
                logp = logits.gather(1, target[:, None]).squeeze(1) - torch.logsumexp(logits, dim=-1)
                lp.append(logp.cpu().numpy())
            probabilities.append(np.concatenate(lp))
            lengths.append(n)
            for (kind, l), a in current.items():
                (hidden if kind == "hidden" else mlp).setdefault(l, []).append(a)
            del result
    for handle in handles:
        handle.remove()
    maximum = max(lengths)
    for kind, store in [("hidden", hidden), ("mlp_act", mlp)]:
        for l, arrays in store.items():
            padded = np.zeros((len(texts), maximum, arrays[0].shape[-1]), np.float16)
            for i, a in enumerate(arrays):
                padded[i, :len(a)] = a
            np.save(out / f"{kind}_{l}.npy", padded)
    padded = np.full((len(texts), maximum-1), np.nan, np.float32)
    for i, a in enumerate(probabilities):
        padded[i, :len(a)] = a
    np.save(out / "token_logprobs.npy", padded)
    write_json(out / "meta.json", {"texts": texts, "lengths": lengths, "model": args.model,
                                   "request_sha256": sha(args.request), "backend": "transformers-sdpa"})
    print("Probe token lengths:", lengths)


def candidate(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    probe_dir = Path(args.probe)
    meta = json.loads((probe_dir / "meta.json").read_text())
    lengths = meta["lengths"]
    if len(lengths) != 8:
        raise ValueError("Expected the four alternating productive/repetitive pairs of fwd_4")
    tensors = load_file(str(checkpoint_path(args.base)))
    report = {"alpha": args.alpha, "topk": args.topk, "mode": args.mode, "layers": {}}
    for layer in args.layers:
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        w = tensors[name].float()
        h = np.load(probe_dir / f"hidden_{layer+1}.npy").astype(np.float32)
        a = np.load(probe_dir / f"mlp_act_{layer}.npy").astype(np.float32)
        hm = np.stack([h[i, max(0, n-36):n].mean(0) for i, n in enumerate(lengths)])
        am = np.stack([a[i, max(0, n-24):n].mean(0) for i, n in enumerate(lengths)])
        u = torch.from_numpy((hm[1::2]-hm[::2]).mean(0))
        u /= u.norm()
        projection = u @ w
        score = projection * torch.from_numpy((am[1::2]-am[::2]).mean(0))
        selected = torch.argsort(score, descending=True)[:args.topk] if args.topk else torch.arange(w.shape[1])
        mask = torch.zeros(w.shape[1]); mask[selected] = 1
        delta = -args.alpha * torch.outer(u, mask*projection)
        if args.mode == "random":
            g = torch.Generator().manual_seed(173 + layer)
            v = torch.randn(u.shape, generator=g)
            v -= u*torch.dot(v, u); v /= v.norm()
            control = torch.outer(v, v @ w)
            delta = -control * (delta.norm()/control.norm())
        changed = (w+delta).to(torch.bfloat16).contiguous()
        tensors[name] = changed
        report["layers"][layer] = {"selected": sorted(selected.tolist()),
            "relative_fp32_change": float(delta.norm()/w.norm()),
            "changed_bf16_entries": int((changed != w.to(torch.bfloat16)).sum())}
    copy_config(args.base, args.out)
    save_file(tensors, str(Path(args.out)/"model.safetensors"), metadata={"format": "pt"})
    write_json(Path(args.out)/"candidate.json", report)
    print(json.dumps(report, indent=2))


def compare(args):
    import numpy as np
    a, b = read_jsonl(args.original), read_jsonl(args.edited)
    x = {(r["id"], r["sample"]): r for r in a}
    y = {(r["id"], r["sample"]): r for r in b}
    if len(x) != len(a) or len(y) != len(b) or set(x) != set(y):
        raise ValueError("Both files must contain exactly the same unique id/sample pairs")
    # Cluster multiple samples by problem: SE over problem means, not an
    # artificially large count of correlated samples from the same question.
    ids = sorted({k[0] for k in x})
    differences = np.array([np.mean([y[k]["correct"]-x[k]["correct"]
                                    for k in x if k[0] == ident]) for ident in ids])
    se = float(differences.std(ddof=1)/np.sqrt(len(ids))) if len(ids)>1 else None
    report = {"problems": len(ids), "samples": len(a),
              "original_accuracy": np.mean([r["correct"] for r in a]).item(),
              "edited_accuracy": np.mean([r["correct"] for r in b]).item(),
              "gain": differences.mean().item(), "paired_problem_se": se,
              "rescues": sum(y[k]["correct"]>x[k]["correct"] for k in x),
              "regressions": sum(y[k]["correct"]<x[k]["correct"] for k in x)}
    print(json.dumps(report, indent=2))
    if args.out:
        write_json(args.out, report)


def transfer(args):
    import numpy as np
    base = Path(args.original); edit = Path(args.edited)
    bm = json.loads((base/"meta.json").read_text())
    em = json.loads((edit/"meta.json").read_text())
    bounds = json.loads(Path(args.bounds).read_text())
    if bm["texts"] != em["texts"] or bm["lengths"] != em["lengths"]:
        raise ValueError("transfer inputs/token lengths differ")
    if [b["length"] for b in bounds] != bm["lengths"]:
        raise ValueError("tokenization differs from recorded boundaries; recompute boundaries before comparing")
    delta = np.load(edit/"token_logprobs.npy") - np.load(base/"token_logprobs.npy")
    report = {}
    for condition in ["productive", "repetitive", "premature_close"]:
        values = []
        for i, b in enumerate(bounds):
            if b["condition"] == condition:
                values.append(float(delta[i, b["length"]-2]) if condition == "premature_close"
                              else float(delta[i, b["start"]-1:b["length"]-1].mean()))
        report[condition] = {"per_text_delta_nats": values, "mean_delta_nats": float(np.mean(values))}
    print(json.dumps(report, indent=2))
    if args.out:
        write_json(args.out, report)


def hardware(args):
    import torch
    print("Python", sys.version, "torch", torch.__version__, "CUDA runtime", torch.version.cuda)
    print("torch compiled architectures", torch.cuda.get_arch_list())
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible")
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i),
              torch.cuda.get_device_properties(i).total_memory/(1<<30), "GiB")
        x = torch.randn((1024,1024),device=f"cuda:{i}",dtype=torch.bfloat16)
        print("BF16 matmul", float((x@x).float().mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("hardware"); q.set_defaults(func=hardware)
    q = sub.add_parser("prepare"); q.add_argument("--out", default="/data"); q.set_defaults(func=prepare)
    q = sub.add_parser("rebuild"); q.add_argument("--base", default="/data/base"); q.add_argument("--out", required=True); q.set_defaults(func=rebuild)
    q = sub.add_parser("select"); q.add_argument("--request", required=True); q.add_argument("--key"); q.add_argument("--out", required=True); q.set_defaults(func=select)
    q = sub.add_parser("generate")
    q.add_argument("--model", required=True); q.add_argument("--rows", required=True); q.add_argument("--out", required=True)
    q.add_argument("--n", type=int, default=1); q.add_argument("--limit", type=int, default=0)
    q.add_argument("--seed", type=int, default=0); q.add_argument("--max-tokens", type=int, default=32768)
    q.add_argument("--max-seqs", type=int, default=8); q.add_argument("--batch-tokens", type=int, default=4096)
    q.add_argument("--memory", type=float, default=.85); q.add_argument("--eager", action="store_true")
    q.add_argument("--queue", type=int, default=64); q.add_argument("--shards", type=int, default=1); q.add_argument("--shard-index", type=int, default=0)
    q.set_defaults(func=generate)
    q = sub.add_parser("probe"); q.add_argument("--model", required=True); q.add_argument("--request", required=True); q.add_argument("--out", required=True); q.set_defaults(func=probe)
    q = sub.add_parser("candidate"); q.add_argument("--base", default="/data/base"); q.add_argument("--probe", required=True)
    q.add_argument("--out", required=True); q.add_argument("--alpha", type=float, default=6)
    q.add_argument("--topk", type=int, default=64); q.add_argument("--layers", nargs="+", type=int, default=[15,19])
    q.add_argument("--mode", choices=["target","random"], default="target"); q.set_defaults(func=candidate)
    q = sub.add_parser("compare"); q.add_argument("--original", required=True); q.add_argument("--edited", required=True); q.add_argument("--out"); q.set_defaults(func=compare)
    q = sub.add_parser("transfer"); q.add_argument("--original", required=True); q.add_argument("--edited", required=True); q.add_argument("--bounds", required=True); q.add_argument("--out"); q.set_defaults(func=transfer)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
