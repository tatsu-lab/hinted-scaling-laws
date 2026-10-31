"""Grade rollouts written by `single_runner.py`, in place.

Fills each row's `grade` ("correct", "incorrect" or "parse_error"), reading the benchmark from
the path unless --benchmark is given. GSM8K and AIME use the benchmark's `verify()`; HLE uses
the judge in `hinted.benchmarks.hle_grader`. Batch files go to `<inputs>/grading/` unless
--manifest-dir is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

BENCHMARKS = ["gsm8k", "aime", "hle"]
LOCAL_GRADER_ID = {"gsm8k": "gsm8k_strict_first_hash",
                   "aime": "aime_marked_integer"}

BATCH_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
CHUNK_SIZE = 40_000            # under OpenAI's 50k-request / 200 MB batch limit
# Per-request token use measured on earlier HLE grading passes.
EST_INPUT_TOKENS, EST_OUTPUT_TOKENS = 741, 131
BATCH_PRICE_PER_1M = {"input": 0.10, "output": 0.625}   # gpt-5.4-nano, Batch
TERMINAL_BAD = {"failed", "expired", "cancelled"}


# ---------------------------------------------------------------------------
# Rollout files
# ---------------------------------------------------------------------------

def find_rollout_files(paths: list[str]) -> list[Path]:
    """Rollout .jsonl files under the given paths, skipping grading artefacts."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*.jsonl") if "grading" not in f.parts]
        else:
            raise SystemExit(f"no such file or directory: {p}")
    return sorted(set(f.resolve() for f in files))


def infer_benchmark(files: list[Path]) -> str:
    """The benchmark named by the `<benchmark>__<family>` path segment."""
    found = set()
    for f in files:
        hits = {part.split("__")[0] for part in f.parts if "__" in part}
        found |= hits & set(BENCHMARKS)
    if len(found) != 1:
        raise SystemExit(
            f"cannot tell the benchmark from the paths (found {sorted(found) or 'none'}); "
            "pass --benchmark, and grade one benchmark per invocation")
    return found.pop()


def read_rows(fp: Path) -> list[dict]:
    with fp.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(fp: Path, rows: list[dict]) -> None:
    """Atomic rewrite: an interrupted pass cannot corrupt the rollout file."""
    tmp = fp.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, fp)


def needs_grade(row: dict, regrade: bool) -> bool:
    if row.get("is_error"):
        return False            # no response to grade
    return regrade or row.get("grade") is None


def rewrite_with(files: list[Path], patches: dict[tuple[int, str], dict]) -> int:
    """Apply {(file index, task_key): fields} to the rollout files."""
    applied = 0
    by_file: dict[int, dict[str, dict]] = {}
    for (i, key), patch in patches.items():
        by_file.setdefault(i, {})[key] = patch
    for i, keyed in by_file.items():
        rows = read_rows(files[i])
        for row in rows:
            patch = keyed.get(row["task_key"])
            if patch is not None:
                row.update(patch)
                applied += 1
        write_rows(files[i], rows)
    return applied


# ---------------------------------------------------------------------------
# AIME / GSM8K
# ---------------------------------------------------------------------------

def grade_local(files: list[Path], benchmark: str, regrade: bool) -> None:
    if benchmark == "gsm8k":
        from hinted.benchmarks.gsm8k_delphi import GSM8KDelphiBenchmark
        bench = GSM8KDelphiBenchmark()
    else:
        from hinted.benchmarks.aime import AIMEBenchmark
        bench = AIMEBenchmark()
    gid = LOCAL_GRADER_ID[benchmark]

    tally = {"correct": 0, "incorrect": 0, "parse_error": 0}
    skipped = 0
    for fp in files:
        rows = read_rows(fp)
        touched = 0
        for row in rows:
            if not needs_grade(row, regrade):
                skipped += 1
                continue
            g = bench.verify(row["response_text"] or "", str(row["answer"])).value
            row["grade"], row["grader_id"] = g, gid
            tally[g] += 1
            touched += 1
        if touched:
            write_rows(fp, rows)
        print(f"  {fp.name}: {touched:,} graded", flush=True)
    print(f"\n{sum(tally.values()):,} graded with {gid}: "
          + " ".join(f"{k}={v:,}" for k, v in tally.items())
          + f"; {skipped:,} skipped (already graded or runner error)")
    if tally["parse_error"]:
        print("parse_error = no extractable answer; count it as incorrect.")


# ---------------------------------------------------------------------------
# HLE: shared pieces
# ---------------------------------------------------------------------------

def load_questions(path: str) -> dict[str, str]:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["problem_id"]: r["question"] for r in rows}


def custom_id(file_index: int, task_key: str) -> str:
    # task_key alone is not unique across methods (a truncate and a mask row
    # share model, problem, level and sample), so name the file too.
    cid = f"{file_index}|{task_key}"
    if len(cid) > 64:
        raise ValueError(f"custom_id longer than OpenAI's 64 chars: {cid!r}")
    return cid


def split_custom_id(cid: str) -> tuple[int, str]:
    i, key = cid.split("|", 1)
    return int(i), key


def iter_judge_work(files: list[Path], questions: dict[str, str],
                    regrade: bool, stats: dict) -> Iterator[tuple[str, dict]]:
    """(custom_id, row) for every row the judge must see."""
    for i, fp in enumerate(files):
        for row in read_rows(fp):
            stats["total"] += 1
            if not needs_grade(row, regrade):
                stats["skipped"] += 1
                continue
            if not row.get("response_text"):
                stats["empty"] += 1       # graded incorrect without a call
                continue
            if row["problem_id"] not in questions:
                raise SystemExit(f"{row['problem_id']} missing from the questions file")
            stats["to_judge"] += 1
            yield custom_id(i, row["task_key"]), row


def empty_response_patches(files: list[Path], regrade: bool, gid: str) -> dict:
    patches = {}
    for i, fp in enumerate(files):
        for row in read_rows(fp):
            if needs_grade(row, regrade) and not row.get("response_text"):
                patches[(i, row["task_key"])] = {"grade": "incorrect",
                                                 "grader_id": gid}
    return patches


def grade_patch(g, gid: str) -> dict:
    """Row fields for one HLEGrade; a failed judge call leaves grade null."""
    if g.is_correct is None:
        return {"grade": None, "grader_id": gid,
                "judge_error": g.error or "unparseable judge response"}
    return {"grade": "correct" if g.is_correct else "incorrect",
            "extracted_answer": g.extracted_answer, "grader_id": gid}


def cost_usd(n: int) -> float:
    return n * (EST_INPUT_TOKENS / 1e6 * BATCH_PRICE_PER_1M["input"]
                + EST_OUTPUT_TOKENS / 1e6 * BATCH_PRICE_PER_1M["output"])


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# HLE: batch build / submit / retrieve
# ---------------------------------------------------------------------------

def hle_build(files, args, mdir: Path) -> None:
    from hinted.benchmarks.hle_grader import (
        HLE_JUDGE_MODEL, HLE_JUDGE_REASONING_EFFORT, build_hle_judge_request,
        hle_grader_id)

    if list(mdir.glob("requests_*.jsonl")) and not args.force:
        raise SystemExit(f"request files already in {mdir}; --force to rebuild")
    for stale in mdir.glob("requests_*.jsonl"):
        stale.unlink()
    questions = load_questions(args.questions)
    stats = {"total": 0, "skipped": 0, "empty": 0, "to_judge": 0}

    chunks: list[list[str]] = [[]]
    for cid, row in iter_judge_work(files, questions, args.regrade, stats):
        if len(chunks[-1]) >= CHUNK_SIZE:
            chunks.append([])
        body = build_hle_judge_request(question=questions[row["problem_id"]],
                                       response=row["response_text"],
                                       correct_answer=str(row["answer"]))
        chunks[-1].append(json.dumps({"custom_id": cid, "method": "POST",
                                      "url": BATCH_ENDPOINT, "body": body}))
    n = stats["to_judge"]
    print(f"rows {stats['total']:,}: {n:,} to judge, {stats['empty']:,} empty "
          f"(graded incorrect on retrieve), {stats['skipped']:,} skipped")
    if not n:
        print("nothing to judge.")
        return

    file_meta = []
    for k, lines in enumerate(chunks):
        rf = mdir / f"requests_{k:03d}.jsonl"
        rf.write_text("\n".join(lines) + "\n")
        file_meta.append({"name": rf.name, "n_requests": len(lines),
                          "sha256": sha256_of(rf)})
    (mdir / "build_manifest.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rollout_files": [str(f) for f in files],
        "questions": args.questions,
        "judge_model": HLE_JUDGE_MODEL,
        "reasoning_effort": HLE_JUDGE_REASONING_EFFORT,
        "grader_id": hle_grader_id(),
        "regrade": args.regrade,
        "stats": stats,
        "est_cost_usd": round(cost_usd(n), 2),
        "request_files": file_meta,
    }, indent=2))
    print(f"wrote {len(file_meta)} request file(s) to {mdir}; "
          f"est. Batch cost ~${cost_usd(n):,.2f}. Nothing submitted.")
    print(f"inspect them, then: --mode submit --manifest-dir {mdir}")


def hle_submit(args, mdir: Path) -> None:
    from openai import OpenAI

    build = json.loads((mdir / "build_manifest.json").read_text())
    batch_path = mdir / "batch_manifest.json"
    if batch_path.exists() and not args.force:
        raise SystemExit(f"{batch_path} exists; --force to resubmit")
    total = sum(f["n_requests"] for f in build["request_files"])
    print(f"submitting {len(build['request_files'])} file(s), {total:,} requests, "
          f"est. ~${cost_usd(total):,.2f}")

    client = OpenAI()
    batches: list[dict] = []
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "grader_id": build["grader_id"], "batches": batches,
                "complete": False}
    for meta in build["request_files"]:
        rf = mdir / meta["name"]
        if sha256_of(rf) != meta["sha256"]:
            print(f"  NOTE {rf.name} changed since build; submitting as it is now")
        up = client.files.create(file=rf.open("rb"), purpose="batch")
        for _ in range(240):                  # wait for the upload to process
            status = client.files.retrieve(up.id).status
            if status == "processed":
                break
            if status in ("error", "failed"):
                raise SystemExit(f"upload {up.id} failed: {status}")
            time.sleep(5)
        else:
            raise SystemExit(f"upload {up.id} not processed after 20 min")
        b = client.batches.create(input_file_id=up.id, endpoint=BATCH_ENDPOINT,
                                  completion_window=COMPLETION_WINDOW)
        batches.append({"requests_file": rf.name, "input_file_id": up.id,
                        "batch_id": b.id})
        # Persist after every batch: one that exists on OpenAI's side but not
        # in this file is paid for and cannot be retrieved.
        batch_path.write_text(json.dumps(manifest, indent=2))
        print(f"  {rf.name}: batch {b.id}", flush=True)
    manifest["complete"] = True
    batch_path.write_text(json.dumps(manifest, indent=2))
    print("submitted. Collect results with --mode retrieve.")


def hle_retrieve(args, mdir: Path) -> None:
    from openai import OpenAI
    from hinted.benchmarks.hle_grader import parse_hle_judge_response

    build = json.loads((mdir / "build_manifest.json").read_text())
    batch = json.loads((mdir / "batch_manifest.json").read_text())
    files = [Path(f) for f in build["rollout_files"]]
    gid = batch["grader_id"]
    raw = mdir / "raw"
    raw.mkdir(exist_ok=True)
    client = OpenAI()

    patches: dict = {}
    pending = []
    for entry in batch["batches"]:
        dump = raw / entry["requests_file"].replace("requests_", "results_")
        if not dump.exists():
            b = client.batches.retrieve(entry["batch_id"])
            if b.status != "completed":
                pending.append((entry["batch_id"], b.status))
                continue
            # Save the replies verbatim before parsing: OpenAI expires output
            # files, and the judge's reasoning and token usage live only here.
            dump.write_text(client.files.content(b.output_file_id).text)
            if getattr(b, "error_file_id", None):
                (raw / dump.name.replace("results_", "errors_")).write_text(
                    client.files.content(b.error_file_id).text)
        for line in dump.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            resp = rec.get("response") or {}
            key = split_custom_id(rec["custom_id"])
            if rec.get("error") or resp.get("status_code") != 200:
                patches[key] = {"grade": None, "grader_id": gid,
                                "judge_error": str(rec.get("error") or resp)[:200]}
            else:
                content = resp["body"]["choices"][0]["message"]["content"]
                patches[key] = grade_patch(
                    parse_hle_judge_response(content, grader_id=gid), gid)

    for bid, status in pending:
        flag = "  <-- will never complete; rebuild and resubmit" if status in TERMINAL_BAD else ""
        print(f"  batch {bid}: {status}{flag}")
    if pending:
        print("some batches are not finished; writing what has arrived.")
    patches.update(empty_response_patches(files, build["regrade"], gid))
    report(rewrite_with(files, patches), patches)


# ---------------------------------------------------------------------------
# HLE: synchronous, for smoke tests
# ---------------------------------------------------------------------------

def hle_sync(files, args) -> None:
    from openai import OpenAI
    from hinted.benchmarks.hle_grader import grade_hle_response, hle_grader_id

    questions = load_questions(args.questions)
    gid = hle_grader_id()
    stats = {"total": 0, "skipped": 0, "empty": 0, "to_judge": 0}
    work = list(iter_judge_work(files, questions, args.regrade, stats))
    print(f"judging {len(work):,} rollouts at full price "
          f"(~${2 * cost_usd(len(work)):,.2f}) with {args.workers} threads")
    client = OpenAI()

    def one(cid_row):
        cid, row = cid_row
        g = grade_hle_response(questions[row["problem_id"]], row["response_text"],
                               str(row["answer"]), client=client)
        return cid, grade_patch(g, gid)

    patches = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, w) for w in work]
        for n, fut in enumerate(as_completed(futures), 1):
            cid, patch = fut.result()
            patches[split_custom_id(cid)] = patch
            if n % 500 == 0:
                print(f"  {n:,}/{len(work):,}", flush=True)
    patches.update(empty_response_patches(files, args.regrade, gid))
    report(rewrite_with(files, patches), patches)


def report(applied: int, patches: dict) -> None:
    grades = [p["grade"] for p in patches.values()]
    print(f"\nwrote {applied:,} grades: correct={grades.count('correct'):,} "
          f"incorrect={grades.count('incorrect'):,} "
          f"judge failures left ungraded={grades.count(None):,}")
    if applied != len(patches):
        print(f"WARNING: {len(patches) - applied:,} results matched no row")


# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()   # OPENAI_API_KEY for the HLE judge
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="rollout files or directories")
    ap.add_argument("--benchmark", choices=BENCHMARKS, default=None,
                    help="default: read from the <benchmark>__<family> path segment")
    ap.add_argument("--mode", choices=["build", "submit", "retrieve", "sync"],
                    default=None, help="HLE only; AIME and GSM8K grade locally")
    ap.add_argument("--regrade", action="store_true",
                    help="grade rows that already have a grade")
    ap.add_argument("--questions", default="data/benchmarks/hle.jsonl")
    ap.add_argument("--manifest-dir", default=None,
                    help="HLE batch artefacts; default <common parent>/grading")
    ap.add_argument("--workers", type=int, default=16, help="--mode sync threads")
    ap.add_argument("--force", action="store_true",
                    help="rebuild existing request files / resubmit")
    args = ap.parse_args()

    files = find_rollout_files(args.paths)
    if not files:
        raise SystemExit("no rollout files found")
    benchmark = args.benchmark or infer_benchmark(files)
    print(f"{benchmark}: {len(files)} rollout file(s)")

    if benchmark != "hle":
        if args.mode:
            raise SystemExit(f"--mode is for HLE; {benchmark} grades locally")
        grade_local(files, benchmark, args.regrade)
        return

    if args.mode is None:
        raise SystemExit("HLE needs --mode: build, submit, retrieve, or sync")
    mdir = (Path(args.manifest_dir) if args.manifest_dir
            else Path(os.path.commonpath([str(f.parent) for f in files])) / "grading")
    mdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "build":
        hle_build(files, args, mdir)
    elif args.mode == "submit":
        hle_submit(args, mdir)
    elif args.mode == "retrieve":
        hle_retrieve(args, mdir)
    else:
        hle_sync(files, args)


if __name__ == "__main__":
    main()
