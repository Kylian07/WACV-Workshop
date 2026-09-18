"""Checks the CLEVR loader against a synthetic file in the real CLEVR format.

The streaming reader exists because CLEVR_train_questions.json is ~800 MB and
json.load on it is a good way to lose a Kaggle session to an OOM.  A reader that
hand-rolls buffering is exactly the kind of code that works on the fixture and
fails on the real file, so the fixture here is deliberately written with the
awkward parts: a preamble before "questions", nested program objects, unicode,
escaped quotes and braces inside a question string, and a chunk size small
enough that objects straddle reads.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trail.config import Config
from trail.data import clevr


def _fixture(n=257):
    qs = []
    for i in range(n):
        depth = 3 + (i % 14)
        prog = [{"function": "scene", "inputs": [], "value_inputs": []}]
        for k in range(depth - 1):
            fn = "relate" if k % 3 == 0 else "filter_color"
            prog.append({"function": fn, "inputs": [k], "value_inputs": ["red"]})
        colour = ["red", "blue", "green", "gray"][i % 4]
        shape = ["cube", "sphere", "cylinder"][i % 3]
        q = f'Is the "{colour}" {shape} {{left}} of the sphere é #{i}?'
        qs.append({
            "image_index": i % 40, "question_index": i, "question": q,
            "answer": ["yes", "no", "3", "cube"][i % 4],
            "image_filename": f"CLEVR_train_{i % 40:06d}.png",
            "question_family_index": i % 7, "split": "train", "program": prog,
        })
    return {"info": {"license": "CC BY 4.0", "version": "1.0", "date": "2/14/2017"},
            "questions": qs}


def test_streaming_reader_matches_json_load():
    data = _fixture()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "CLEVR_train_questions.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        got = list(clevr.iter_questions(p))
        assert len(got) == len(data["questions"]), (len(got), len(data["questions"]))
        for a, b in zip(got, data["questions"]):
            assert a["question"] == b["question"]
            assert a["answer"] == b["answer"]
            assert a["image_index"] == b["image_index"]
            assert a["program"] == [x["function"] for x in b["program"]]
        assert len(list(clevr.iter_questions(p, limit=17))) == 17
    print(f"streaming reader ({len(got)} questions, unicode + braces + escapes): OK")


def test_program_depth():
    d, nrel = clevr.program_depth(["scene", "filter_color", "relate", "same_shape", "query_color"])
    assert (d, nrel) == (5, 2), (d, nrel)
    assert clevr.program_depth(None) == (0, 0)
    print("program depth / relational-node count: OK")


def test_splits_are_disjoint_and_nonempty():
    data = _fixture(400)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "questions").mkdir()
        (root / "questions" / "CLEVR_train_questions.json").write_text(json.dumps(data))
        (root / "questions" / "CLEVR_val_questions.json").write_text(json.dumps(data))

        cfg = Config(split="depth", depth_train_max=8, depth_test_min=12,
                     max_train_questions=0, max_val_questions=0)
        tr = clevr.build_index(root, cfg, "train")
        va = clevr.build_index(root, cfg, "val")
        assert tr and va, (len(tr), len(va))
        assert max(r["depth"] for r in tr) <= 8
        assert min(r["depth"] for r in va) >= 12
        assert not ({r["question"] for r in tr} & {r["question"] for r in va})
        n_tr, n_va = len(tr), len(va)

        cfg = Config(split="hoc", hoc_heldout=("red", "cube"))
        tr = clevr.build_index(root, cfg, "train")
        va = clevr.build_index(root, cfg, "val")
        assert tr and va, "both sides of the hoc split must be non-empty"
        assert all(not ("red" in r["question"].lower() and "cube" in r["question"].lower())
                   for r in tr)
        assert all("red" in r["question"].lower() and "cube" in r["question"].lower()
                   for r in va)
        print(f"depth split ({n_tr} train / {n_va} test) and held-out-combination "
              f"split ({len(tr)} train / {len(va)} test): OK")


def test_vocab_and_discovery():
    data = _fixture(50)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "questions").mkdir()
        (root / "questions" / "CLEVR_train_questions.json").write_text(json.dumps(data))
        assert clevr.find_clevr_root(str(root)) == root
        # discovery should also work one level of nesting down, as Kaggle mirrors nest
        nested = Path(d) / "outer"
        (nested / "CLEVR_v1.0" / "questions").mkdir(parents=True)
        (nested / "CLEVR_v1.0" / "questions" / "CLEVR_train_questions.json").write_text(
            json.dumps(data))
        assert clevr.find_clevr_root(str(nested)).name == "CLEVR_v1.0"

        recs = clevr.build_index(root, Config(), "train")
        stoi, atoi = clevr.build_vocab(recs)
        assert stoi["<pad>"] == 0 and stoi["<unk>"] == 1
        assert set(atoi) == {"yes", "no", "3", "cube"}
        print(f"root discovery (flat + nested) and vocab ({len(stoi)} words, {len(atoi)} answers): OK")


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL:
        fn()
    print(f"\nAll {len(ALL)} CLEVR I/O checks passed.")
