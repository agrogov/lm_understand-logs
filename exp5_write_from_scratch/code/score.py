"""Read a reply's edits block into a network and score it on MinAtar, as described in the prompts.

    python score.py <game> <reply answer.md | network .npz> [--episodes 1000] [--seed 0] [--save out.npz]

<game>: breakout, seaquest or asterix. A reply is read the way the prompt says: only the last edits block,
statements in order, a statement that cannot be read is skipped. --save writes the network read from a reply in the
same format as networks/<game>/hand.npz.

Scoring follows "HOW YOUR NETWORK IS SCORED": MinAtar 1.0.15 with sticky actions 0.1, actions sampled from the
softmax of the logits, at most 10,000 steps per episode. Prints one JSON line with the fields of eval/minatar_eval.json.

Requires numpy and minatar==1.0.15.
"""
import argparse
import ast
import json
import re

import numpy as np
from minatar import Environment

# the 6 input channels <- MinAtar's own channels, from each prompt's THE INPUT section
CHANNELS = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "seaquest": [["sub_front", "sub_back"], ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 ["enemy_fish", "enemy_sub", "diver"], ["friendly_bullet"], ["trail", "sub_back"],
                 ["oxygen_guage", "diver_guage"]],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
}
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
NPZ_KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
            "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
STATEMENT = re.compile(r"^(W1|b1|W2|b2|W3|b3)\s*(?:\[([^\]]*)\])?\s*([-+*]?=)\s*(.+)$", re.S)


def last_edits_block(text):
    starts = [m.end() for m in re.finditer(r"```edits[^\n]*\n", text)]
    if not starts:
        return None, False
    body = text[starts[-1]:]
    end = body.find("```")
    return (body, True) if end < 0 else (body[:end], False)


def split_statements(block):
    # a statement runs until its brackets close, so a long list may continue over lines
    out, cur = [], ""
    for line in block.split("\n"):
        line = line.split("#", 1)[0]
        if not line.strip() and not cur:
            continue
        cur += " " + line
        if cur.count("[") == cur.count("]") and "=" in cur:
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_index(s):
    idx = []
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            idx.append(slice(*[int(b) if b.strip() else None for b in part.split(":")]))
        else:
            idx.append(int(part))
    return tuple(idx)


def read_reply(text):
    w = {k: np.zeros(s, dtype=np.float32) for k, s in SHAPES.items()}
    block, cut_off = last_edits_block(text)
    report = {"block": block is not None, "statements": 0, "applied": 0, "errors": [], "cut_off": cut_off}
    for st in split_statements(block or ""):
        report["statements"] += 1
        try:
            name, idx, op, val = STATEMENT.match(st).groups()
            v = np.array(ast.literal_eval(val.strip()), dtype=np.float64)
            if np.any(np.abs(v) > 1e6):
                raise ValueError("value above 1e6")
            ix = parse_index(idx) if idx is not None else (slice(None),)
            t = w[name]
            if op == "=":
                t[ix] = v
            elif op == "+=":
                t[ix] = t[ix] + v
            elif op == "-=":
                t[ix] = t[ix] - v
            else:
                t[ix] = t[ix] * v
            report["applied"] += 1
        except Exception as e:
            report["errors"].append(f"{st[:80]}: {e!r}")
    return w, report


def observe(env, game):
    s = env.state()
    x = np.zeros((10, 10, 6), dtype=np.float32)
    for c, sources in enumerate(CHANNELS[game]):
        for name in sources:
            x[:, :, c] = np.maximum(x[:, :, c], s[:, :, env.env.channels[name]])
    return x.reshape(-1)


def score(game, w, episodes, seed):
    env = Environment(game, sticky_action_prob=0.1)
    env.seed(seed)
    rng = np.random.default_rng(seed + 1)
    returns, lengths = [], []
    for _ in range(episodes):
        env.reset()
        env.last_action = 0
        total, t, done = 0.0, 0, False
        while not done and t < 10000:
            h1 = np.tanh(observe(env, game) @ w["W1"] + w["b1"])
            h2 = np.tanh(h1 @ w["W2"] + w["b2"])
            logits = (h2 @ w["W3"] + w["b3"]).astype(np.float64)
            p = np.exp(logits - logits.max())
            r, done = env.act(int(rng.choice(6, p=p / p.sum())))
            total += r
            t += 1
        returns.append(total)
        lengths.append(t)
    r = np.array(returns)
    return {"mean": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r))), "episodes": len(r),
            "max": float(r.max()), "mean_length": float(np.mean(lengths))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=sorted(CHANNELS))
    ap.add_argument("path")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save")
    a = ap.parse_args()
    out = {}
    if a.path.endswith(".npz"):
        z = np.load(a.path)
        w = {k: z[v].astype(np.float32) for k, v in NPZ_KEYS.items()}
    else:
        w, report = read_reply(open(a.path).read())
        out["edit_report"] = {**report, "errors": report["errors"][:20]}
        if a.save:
            np.savez(a.save, **{NPZ_KEYS[k]: v for k, v in w.items()})
    out["nonzero"] = int(sum((v != 0).sum() for v in w.values()))
    out.update(score(a.game, w, a.episodes, a.seed))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
