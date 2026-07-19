#!/usr/bin/env python
"""
review_questions.py — Phase 2 deliverable (human review pass)

Interactive CLI for reviewing the AI-drafted question/gold-passage dataset in
data/eval/questions_draft.csv. Shows each question next to the full text of
its linked chunk(s) so you can quickly approve, edit, or reject — this is
what turns the draft into real gold-passage ground truth for Recall@K/MRR.

Usage
-----
  python review_questions.py                    # review all draft_needs_review rows
  python review_questions.py --include-blocked  # also show blocked_no_corpus rows
  python review_questions.py --redo-rejected    # re-review previously rejected rows

Progress is saved after every single decision, so you can quit ('q') anytime
and resume later — already-approved/rejected rows are skipped automatically.

Controls (per question)
------------------------
  a  approve as-is
  e  edit the question text, then approve
  r  reject (asks for a short reason, stored in notes)
  s  skip for now (leave status unchanged, revisit later)
  f  show the FULL untruncated chunk text before deciding
  q  save progress and quit
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "eval" / "questions_draft.csv"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
FIELDNAMES = ["question_id", "question", "intent", "framework", "gold_chunk_ids",
              "gold_sections", "source_titles", "status", "notes"]

PREVIEW_CHARS = 1200
FULL_CAP_CHARS = 6000  # even in "full" mode, cap huge chunks so the terminal stays usable
CONTROLS = "[a]pprove  [e]dit+approve  [r]eject  [s]kip-to-next  [f]ull text (same question)  [q]uit"


def load_chunks():
    chunks = {}
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return chunks


def load_questions():
    with open(QUESTIONS_PATH, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_questions(rows):
    with open(QUESTIONS_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def print_row(row, chunks, full=False):
    print("\n" + "=" * 78)
    print(f"{row['question_id']}  [{row['framework']}]  intent={row['intent']}")
    print("-" * 78)
    print("Q:", row["question"])
    ids = [i for i in row["gold_chunk_ids"].split(";") if i]
    if not ids:
        print("\n(no gold chunk linked -- documented out-of-corpus gap)")
        print("notes:", row["notes"])
        return
    for cid in ids:
        c = chunks.get(cid)
        if not c:
            print(f"\n!! chunk {cid} not found in chunks.jsonl (corpus may have changed since this draft)")
            continue
        print(f"\n--- gold chunk: {cid} | {c['title']} | sec {c['section']} ---")
        text = c["text"]
        cap = FULL_CAP_CHARS if full else PREVIEW_CHARS
        if len(text) > cap:
            note = "still truncated -- see chunks.jsonl for the rest" if full else "press 'f' to see more"
            print(text[:cap] + f"\n...[{note}; {len(text)} chars total]")
        else:
            print(text)


def main():
    ap = argparse.ArgumentParser(description="Interactive review of drafted questions")
    ap.add_argument("--include-blocked", action="store_true", help="also review blocked_no_corpus rows")
    ap.add_argument("--redo-rejected", action="store_true", help="re-review previously rejected rows")
    args = ap.parse_args()

    rows = load_questions()
    chunks = load_chunks()

    targets = {"draft_needs_review"}
    if args.include_blocked:
        targets.add("blocked_no_corpus")
    if args.redo_rejected:
        targets.add("rejected")

    todo = [r for r in rows if r["status"] in targets]
    print(f"{len(todo)} rows to review. Current status counts: {dict(Counter(r['status'] for r in rows))}")

    reviewed = 0
    for i, row in enumerate(todo, start=1):
        full = False
        while True:
            print_row(row, chunks, full=full)
            print(f"\n[{i}/{len(todo)}] {CONTROLS}")
            choice = input("> ").strip().lower()
            if choice == "f":
                full = True
                continue
            if choice == "a":
                row["status"] = "approved"
                reviewed += 1
                break
            if choice == "e":
                new_q = input("New question text: ").strip()
                if new_q:
                    row["question"] = new_q
                row["status"] = "approved"
                row["notes"] = (row["notes"] + " | edited during review").strip(" |")
                reviewed += 1
                break
            if choice == "r":
                reason = input("Reason for rejecting (short): ").strip()
                row["status"] = "rejected"
                row["notes"] = (row["notes"] + f" | rejected: {reason}").strip(" |")
                reviewed += 1
                break
            if choice == "s":
                break
            if choice == "q":
                save_questions(rows)
                print(f"\nSaved. Reviewed {reviewed} this session.")
                print(f"Status counts now: {dict(Counter(r['status'] for r in rows))}")
                return
            print("Unrecognized input -- use a/e/r/s/f/q")
        save_questions(rows)  # persist after every decision, not just on quit

    print(f"\nAll caught up. Reviewed {reviewed} this session.")
    print(f"Status counts now: {dict(Counter(r['status'] for r in rows))}")


if __name__ == "__main__":
    main()
