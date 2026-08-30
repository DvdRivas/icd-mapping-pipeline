"""
icd10-lookup.py — reverse process of icd-experiment.py (ICD-10).

Two phases over each run's CSV:

  1. Reverse lookup, deterministic and offline: icd10_code -> official title,
     written to icd10_lookup. NF codes resolve against nf_dictionary.json.
  2. Consistency, LLM: does that title encompass the original diagnosis?
     yes / no, written to consistency.

Titles come from the same ClaML catalogue that assigned the codes, which
guarantees the two phases agree and removes the network entirely — the WHO
ICD-10 API has no /codeinfo (404) and answers in English only.

The judge sees the title of `category`, the level the pipeline actually
reports, so the title is always broader than the diagnosis by construction.
The prompt says so explicitly; otherwise the model would mark everything as
inconsistent for lack of specificity.

Configuration lives in the CONFIG block below. The only command-line flag is
--overwrite, since redoing already-processed rows is a per-invocation choice.

Usage:
    python icd10-lookup.py [--overwrite]
"""

import argparse
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lmstudio as lms
from tqdm import tqdm

import icd_common as common
from icd10_index import CLAML_FILENAME, ICD10Index
from run_logger import RunLogger

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "medgemma-27b-it"
LMS_HOST = os.environ.get("LMS_HOST", "10.8.0.45:1234")

TEMPERATURE = 0.0          # greedy; see icd-experiment.py for the rationale
PREDICTION_CONFIG = {"temperature": TEMPERATURE}

WORKERS = 8                # threads resolving codes
JUDGE_BATCH = 10           # rows judged concurrently
N_RUNS = 10
MAX_ROWS = None            # set to an int to process only the first N rows

SKIP_CONSISTENCY = False   # True = reverse lookup only, no LLM
EVAL_NF = False            # True = also judge NF rows (see below)

CLAML_PATH: str | None = None

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "lookup-runs.log"     # separate from the experiment log
LOGGING_ENABLED = True
NF_DICT_PATH = SCRIPT_DIR / "nf_dictionary.json"

COLUMNS = ["diagnosis_es", "diagnosis_en", "clinical_summary",
           "icd10_code", "icd_code_complete", "category", "chapter",
           "hierarchical_distance", "hierarchy_path", "mapping_relation",
           "icd10_lookup", "consistency"]


CONSISTENCY_PROMPT = """You are a clinical coding auditor.

You receive a clinical DIAGNOSIS and the OFFICIAL TITLE of the ICD-10
category assigned to it. Answer a single question:

    Is the category title consistent with the diagnosis?

Important context: the codes were truncated to category level, so the title
will always be BROADER than the diagnosis. That is NOT an error.

Answer "yes" when:
- The category correctly encompasses the diagnosis, even if it is broader.
  E.g. diagnosis "AL amyloidosis" / title "Amyloidosis" -> yes
  E.g. diagnosis "Hepatocellular carcinoma" / title "Malignant neoplasm of
      liver and intrahepatic bile ducts" -> yes
- The title is a synonym or terminological variant of the diagnosis.
- The title names the same clinical entity under another denomination.

Answer "no" when:
- The category belongs to a different system, organ or pathological process.
  E.g. diagnosis "Renal malakoplakia" / title "Diseases of oesophagus" -> no
- The category shares words with the diagnosis but designates another entity.
  E.g. diagnosis "Q fever" / title "Yellow fever" -> no
- The title is so unspecific that it carries no real classification.
  E.g. title "Other specified conditions" or "Other disorders" -> no

Answer with ONE word only: yes or no.
No quotes, no punctuation, no explanations."""


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — reverse lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_nf_labels() -> dict:
    """nf_dictionary.json inverted to {code: label} to resolve NF rows."""
    if not NF_DICT_PATH.exists():
        return {}
    nf_dict = json.loads(NF_DICT_PATH.read_text(encoding="utf-8"))
    return {entry["code"]: entry.get("label", "")
            for entry in nf_dict.values() if entry.get("code")}


def resolve_row(row: dict, index: ICD10Index, nf_labels: dict) -> str:
    """
    Fill icd10_lookup for one row; return the outcome for the statistics.

    Titles prefer ClaML's long form, which carries the parent context
    ("Malignant neoplasm: Liver cell carcinoma" rather than just "Liver cell
    carcinoma") — the extra context is what the auditor needs.
    """
    code = (row.get("icd10_code") or "").strip()
    if not code:
        row["icd10_lookup"] = ""
        return "no_code"

    if code.startswith("NF-"):
        label = nf_labels.get(code, "")
        row["icd10_lookup"] = f"[NF] {label}" if label else "[NF] no label"
        return "nf"

    if title := index.title(code):
        row["icd10_lookup"] = title
        return "resolved"

    row["icd10_lookup"] = ""
    return "unresolved"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — consistency
# ─────────────────────────────────────────────────────────────────────────────

async def judge_row(diagnosis: str, lookup: str, model) -> str:
    """"yes" / "no", or "" when the model errors (row left unevaluated)."""
    raw = ""
    try:
        raw = await common._ask(
            model, CONSISTENCY_PROMPT,
            f'DIAGNOSIS: "{diagnosis}"\nICD-10 TITLE: "{lookup}"\n\n'
            f'Is the title consistent with the diagnosis? Answer yes or no.',
            PREDICTION_CONFIG)
        raw = "".join(c for c in raw.lower() if c.isalpha())
        if raw.startswith("yes"):
            return "yes"
        if raw.startswith("no"):
            return "no"
        print(f"\n    [LLM-consistency] Unexpected answer: {raw[:60]!r}")
    except Exception as e:
        print(f"\n    [LLM-consistency] Error: {e} | raw: {raw[:100]}")
    return ""


async def judge_rows(rows: list, model) -> dict:
    """
    Evaluate diagnosis vs icd10_lookup, writing `consistency` in place.

    Unlike icd-experiment.py there is nothing to reset between runs: the judge
    keeps no per-diagnosis cache, so every row really calls the LLM every run.
    """
    stats = {"yes": 0, "no": 0, "skipped_nf": 0, "no_lookup": 0, "error": 0}
    pending = []

    for row in rows:
        code = (row.get("icd10_code") or "").strip()
        lookup = (row.get("icd10_lookup") or "").strip()

        # NF rows have no real labelling to audit: their lookup is the
        # diagnosis itself, so judging them would always yield "yes".
        if code.startswith("NF-") and not EVAL_NF:
            row["consistency"] = ""
            stats["skipped_nf"] += 1
        elif code and not lookup:
            row["consistency"] = "no"      # resolved to nothing, cannot agree
            stats["no_lookup"] += 1
        elif not lookup or not (row.get("diagnosis_en") or "").strip():
            row["consistency"] = ""
        else:
            pending.append(row)

    if not pending:
        print("[LLM] No rows to evaluate.\n")
        return stats

    with tqdm(total=len(pending), desc="Evaluating consistency", unit="row") as bar:
        for start in range(0, len(pending), JUDGE_BATCH):
            chunk = pending[start:start + JUDGE_BATCH]
            verdicts = await asyncio.gather(*(
                judge_row((r.get("diagnosis_en") or "").strip(),
                          (r.get("icd10_lookup") or "").strip(), model)
                for r in chunk))
            for row, verdict in zip(chunk, verdicts):
                row["consistency"] = verdict
                stats[verdict if verdict in ("yes", "no") else "error"] += 1
            bar.update(len(chunk))

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _select(rows: list, column: str, overwrite: bool) -> list:
    """Rows holding a code whose `column` is still empty, unless overwriting."""
    candidates = rows if MAX_ROWS is None else rows[:MAX_ROWS]
    return [r for r in candidates
            if (r.get("icd10_code") or "").strip()
            and (overwrite or not (r.get(column) or "").strip())]


async def process_csv(csv_path: str, index: ICD10Index, model,
                      overwrite: bool) -> None:
    rows, fieldnames = common.read_csv_robust(csv_path, COLUMNS)
    if not rows:
        print(f"[CSV] {csv_path} has no data rows.\n")
        return

    nf_labels = load_nf_labels()
    print(f"[NF]  Dictionary loaded: {len(nf_labels)} codes\n")

    pending = _select(rows, "icd10_lookup", overwrite)
    with_code = sum(1 for r in rows if (r.get("icd10_code") or "").strip())
    print(f"[CSV] {len(rows)} rows | {with_code} with icd10_code | "
          f"{len(pending)} to resolve")
    if not overwrite and with_code > len(pending):
        print(f"[CSV] {with_code - len(pending)} already had a lookup "
              f"(use --overwrite to redo them)")
    print()

    stats = {"resolved": 0, "nf": 0, "unresolved": 0, "no_code": 0}
    if pending:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            outcomes = list(tqdm(
                pool.map(lambda r: resolve_row(r, index, nf_labels), pending),
                total=len(pending), desc="Resolving codes", unit="row"))
        for outcome in outcomes:
            stats[outcome] += 1
    else:
        print("[CSV] Reverse lookup: nothing to do.\n")

    judge_stats = None
    if not SKIP_CONSISTENCY:
        to_judge = _select(rows, "consistency", overwrite)
        if to_judge:
            print(f"\n[LLM] {len(to_judge)} rows to evaluate\n")
            judge_stats = await judge_rows(to_judge, model)
        else:
            print("\n[LLM] Consistency: nothing to do "
                  "(use --overwrite to re-evaluate).\n")

    common.write_csv(csv_path, rows, fieldnames)
    _print_stats(len(pending), stats, judge_stats)


def _print_stats(processed: int, stats: dict, judge_stats: dict | None) -> None:
    if processed:
        divisor = max(processed, 1)
        print("── Reverse lookup statistics ───────────────────────────")
        print(f"  Processed                  : {processed}")
        for key, label in (("resolved", "Resolved (catalogue)"),
                           ("nf", "NF (local dictionary)"),
                           ("unresolved", "Unresolved")):
            print(f"  {label:<27}: {stats[key]} "
                  f"({stats[key] / divisor * 100:.1f}%)")
        print("────────────────────────────────────────────────────────\n")
        if stats["unresolved"]:
            print(f"[!] {stats['unresolved']} codes could not be resolved — "
                  f"usually codes that do not exist as an entity of their own.\n")

    if judge_stats:
        evaluated = judge_stats["yes"] + judge_stats["no"]
        divisor = max(evaluated, 1)
        print("── Consistency diagnosis_en vs icd10_lookup ────────────")
        print(f"  Evaluated by the LLM       : {evaluated}")
        print(f"  Consistent    (yes)        : {judge_stats['yes']} "
              f"({judge_stats['yes'] / divisor * 100:.1f}%)")
        print(f"  Inconsistent  (no)         : {judge_stats['no']} "
              f"({judge_stats['no'] / divisor * 100:.1f}%)")
        for key, label in (("no_lookup", 'Marked "no", no lookup'),
                           ("skipped_nf", "Skipped for being NF"),
                           ("error", "LLM errors")):
            if judge_stats[key]:
                print(f"  {label:<27}: {judge_stats[key]}")
        print("────────────────────────────────────────────────────────\n")


def find_run_csvs() -> list[str]:
    paths = [SCRIPT_DIR / f"run{i}.csv" for i in range(1, N_RUNS + 1)]
    if missing := [p.name for p in paths if not p.exists()]:
        print(f"[CSV] Notice: not found {missing}")
    return [str(p) for p in paths if p.exists()]


async def run_lookup(csv_paths: list[str], index: ICD10Index, overwrite: bool,
                     logger: RunLogger) -> None:
    if SKIP_CONSISTENCY:
        for position, csv_path in enumerate(csv_paths, 1):
            logger.run_header(position, len(csv_paths), csv_path)
            await process_csv(csv_path, index, None, overwrite)
        return

    async with lms.AsyncClient(LMS_HOST) as client:
        print(f"[LLM] Host: {LMS_HOST}\n[LLM] Connecting to: {MODEL_NAME}")
        try:
            model = await client.llm.model(MODEL_NAME)
        except Exception as e:
            print(f"[LLM] ERROR connecting to {LMS_HOST}: {e}")
            print("[LLM] Check LM Studio is serving on that IP:port.\n")
            return
        print("[LLM] Connected\n")

        for position, csv_path in enumerate(csv_paths, 1):
            logger.run_header(position, len(csv_paths), csv_path)
            await process_csv(csv_path, index, model, overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICD-10 reverse lookup: code -> official title, then "
                    "consistency against the diagnosis.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo rows that already have icd10_lookup/consistency")
    args = parser.parse_args()

    csv_paths = find_run_csvs()
    if not csv_paths:
        print(f"[ERROR] No run{{1..{N_RUNS}}}.csv in {SCRIPT_DIR}")
        return
    print(f"[CSV] Iterative execution: {len(csv_paths)} runs found\n")

    index = ICD10Index.load(CLAML_PATH)
    if index is None:
        print(f"[ERROR] {CLAML_FILENAME} not found near {SCRIPT_DIR}. "
              f"There is no way to resolve the codes.")
        return
    print(f"[ICD-10] Catalogue: {len(index)} classes | version {index.version}")
    print(f"[ICD-10] Source: {index.source}\n")

    with RunLogger(LOG_PATH, enabled=LOGGING_ENABLED) as logger:
        logger.session_header({
            "phase": "reverse lookup + consistency",
            "icd version": "ICD-10",
            "model": "(skipped)" if SKIP_CONSISTENCY else MODEL_NAME,
            "lms host": LMS_HOST,
            "temperature": TEMPERATURE,
            "runs": len(csv_paths),
            "rows per run": MAX_ROWS if MAX_ROWS is not None else "all",
            "workers": WORKERS,
            "overwrite": args.overwrite,
            "eval NF rows": EVAL_NF,
            "catalogue": f"{len(index)} classes, version {index.version}",
            "catalogue source": index.source,
        })
        asyncio.run(run_lookup(csv_paths, index, args.overwrite, logger))


if __name__ == "__main__":
    main()
