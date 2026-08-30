"""
icd11-lookup.py — reverse process of icd-experiment.py (ICD-11).

Two phases over each run's CSV:

  1. Reverse lookup, deterministic and offline: icd11_code -> official title,
     written to icd11_lookup. NF codes resolve against nf_dictionary.json.
  2. Consistency, LLM: does that title encompass the original diagnosis?
     yes / no, written to consistency.

Titles come from the WHO ICD-11 API: /codeinfo yields the entity URI, which
is then followed for its official title, with a /search lookup restricted to
the code as fallback. Unlike the ICD-10 folder this phase needs the network.

The judge sees the title of `category`, the level the pipeline actually
reports, so the title is always broader than the diagnosis by construction.
The prompt says so explicitly; otherwise the model would mark everything as
inconsistent for lack of specificity.

Configuration lives in the CONFIG block below. The only command-line flag is
--overwrite, since redoing already-processed rows is a per-invocation choice.

Usage:
    python icd11-lookup.py [--overwrite]
"""

import argparse
import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

import lmstudio as lms
from tqdm import tqdm

import requests
import urllib3

import icd_common as common
from run_logger import RunLogger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

WHO_CLIENT_ID = os.environ.get(
    "WHO_CLIENT_ID",
    "00da56ea-0fe1-465e-b830-61cb0add2173_2741cb3a-0cd6-4977-af95-6258be8bd99a")
WHO_CLIENT_SECRET = os.environ.get(
    "WHO_CLIENT_SECRET", "h1MrlyMGBnGt7Q6kAnSpq8/1s18FkkzPboT7MIaim7o=")

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "lookup-runs.log"     # separate from the experiment log
LOGGING_ENABLED = True
NF_DICT_PATH = SCRIPT_DIR / "nf_dictionary.json"

COLUMNS = ["diagnosis_es", "diagnosis_en", "clinical_summary",
           "icd11_code", "icd_code_complete", "category", "chapter",
           "hierarchical_distance", "hierarchy_path", "mapping_relation",
           "icd11_lookup", "consistency"]


CONSISTENCY_PROMPT = """You are a clinical coding auditor.

You receive a clinical DIAGNOSIS and the OFFICIAL TITLE of the ICD-11
category assigned to it. Answer a single question:

    Is the category title consistent with the diagnosis?

Important context: the codes were truncated to category level, so the title
will always be BROADER than the diagnosis. That is NOT an error.

Answer "yes" when:
- The category correctly encompasses the diagnosis, even if it is broader.
  E.g. diagnosis "AL amyloidosis" / title "Amyloidosis" -> yes
  E.g. diagnosis "Hepatocellular carcinoma" / title "Malignant neoplasms of
      liver or intrahepatic bile ducts" -> yes
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
# WHO ICD-11 API — code -> official title
# ─────────────────────────────────────────────────────────────────────────────

class WHOLookupClient:
    """
    Resolves an ICD-11 code to its official title.

    /codeinfo returns the entity URI (`stemId`), which is then followed for
    the title; a /search restricted to the code covers the cases where
    /codeinfo has no entry. Titles are requested in English, matching
    diagnosis_en and the rest of the pipeline.

    Results are memoised per code: that is deterministic reference data, so
    unlike a per-diagnosis cache it cannot couple one run to the next.
    """

    TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
    BASE_URL = "https://id.who.int/icd/release/11/2024-01/mms"
    CODEINFO_URL = f"{BASE_URL}/codeinfo"
    SEARCH_URL = f"{BASE_URL}/search"
    LANG = "en"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.session = requests.Session()
        self._titles: dict[str, str] = {}
        self._lock = Lock()

    def authenticate(self) -> bool:
        if not self.client_id or not self.client_secret:
            print("[WHO] No credentials. Set WHO_CLIENT_ID and "
                  "WHO_CLIENT_SECRET.\n")
            return False
        try:
            response = self.session.post(
                self.TOKEN_URL,
                data={"client_id": self.client_id,
                      "client_secret": self.client_secret,
                      "scope": "icdapi_access",
                      "grant_type": "client_credentials"},
                verify=False, timeout=15)
            response.raise_for_status()
            self.token = response.json()["access_token"]
            print("[WHO] Token obtained successfully\n")
            return True
        except Exception as e:
            print(f"[WHO] Authentication error: {e}\n")
            return False

    def _get(self, url: str, **kwargs):
        return self.session.get(
            url, headers={"Authorization": f"Bearer {self.token}",
                          "Accept": "application/json",
                          "Accept-Language": self.LANG,
                          "API-Version": "v2"},
            verify=False, timeout=15, **kwargs)

    @staticmethod
    def _title(payload: dict) -> str:
        title = payload.get("title")
        if isinstance(title, dict):
            title = title.get("@value", "")
        return re.sub(r"</?em[^>]*>", "", title or "").strip()

    def _via_codeinfo(self, code: str) -> str:
        response = self._get(f"{self.CODEINFO_URL}/{code}")
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        stem_id = (response.json() or {}).get("stemId", "")
        if not stem_id:
            return ""
        entity = self._get(stem_id)
        if entity.status_code == 404:
            return ""
        entity.raise_for_status()
        return self._title(entity.json())

    def _via_search(self, code: str) -> str:
        """Fallback: search the code as text and take the exact match."""
        response = self._get(self.SEARCH_URL,
                             params={"q": code, "useFlexisearch": "false",
                                     "flatResults": "true",
                                     "highlightingEnabled": "false",
                                     "includeKeywordResult": "false"})
        response.raise_for_status()
        for entity in (response.json().get("destinationEntities") or []):
            if (entity.get("theCode") or "").strip().upper() == code.upper():
                return self._title(entity)
        return ""

    def title(self, code: str) -> str:
        """Official title, or "" when the code could not be resolved."""
        code = (code or "").strip()
        if not code or not self.token:
            return ""
        with self._lock:
            if code in self._titles:
                return self._titles[code]

        title = ""
        try:
            title = self._via_codeinfo(code) or self._via_search(code)
        except Exception as e:
            print(f"\n    [WHO-lookup] {code}: {e}")

        with self._lock:
            self._titles[code] = title
        return title


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


def resolve_row(row: dict, who: WHOLookupClient, nf_labels: dict) -> str:
    """
    Fill icd11_lookup for one row; return the outcome for the statistics.

    """
    code = (row.get("icd11_code") or "").strip()
    if not code:
        row["icd11_lookup"] = ""
        return "no_code"

    if code.startswith("NF-"):
        label = nf_labels.get(code, "")
        row["icd11_lookup"] = f"[NF] {label}" if label else "[NF] no label"
        return "nf"

    if title := who.title(code):
        row["icd11_lookup"] = title
        return "resolved"

    row["icd11_lookup"] = ""
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
            f'DIAGNOSIS: "{diagnosis}"\nICD-11 TITLE: "{lookup}"\n\n'
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
    Evaluate diagnosis vs icd11_lookup, writing `consistency` in place.

    Unlike icd-experiment.py there is nothing to reset between runs: the judge
    keeps no per-diagnosis cache, so every row really calls the LLM every run.
    """
    stats = {"yes": 0, "no": 0, "skipped_nf": 0, "no_lookup": 0, "error": 0}
    pending = []

    for row in rows:
        code = (row.get("icd11_code") or "").strip()
        lookup = (row.get("icd11_lookup") or "").strip()

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
                          (r.get("icd11_lookup") or "").strip(), model)
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
            if (r.get("icd11_code") or "").strip()
            and (overwrite or not (r.get(column) or "").strip())]


async def process_csv(csv_path: str, who: WHOLookupClient, model,
                      overwrite: bool) -> None:
    rows, fieldnames = common.read_csv_robust(csv_path, COLUMNS)
    if not rows:
        print(f"[CSV] {csv_path} has no data rows.\n")
        return

    nf_labels = load_nf_labels()
    print(f"[NF]  Dictionary loaded: {len(nf_labels)} codes\n")

    pending = _select(rows, "icd11_lookup", overwrite)
    with_code = sum(1 for r in rows if (r.get("icd11_code") or "").strip())
    print(f"[CSV] {len(rows)} rows | {with_code} with icd11_code | "
          f"{len(pending)} to resolve")
    if not overwrite and with_code > len(pending):
        print(f"[CSV] {with_code - len(pending)} already had a lookup "
              f"(use --overwrite to redo them)")
    print()

    stats = {"resolved": 0, "nf": 0, "unresolved": 0, "no_code": 0}
    if pending:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            outcomes = list(tqdm(
                pool.map(lambda r: resolve_row(r, who, nf_labels), pending),
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
        print("── Consistency diagnosis_en vs icd11_lookup ────────────")
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


async def run_lookup(csv_paths: list[str], who: WHOLookupClient,
                     overwrite: bool, logger: RunLogger) -> None:
    if SKIP_CONSISTENCY:
        for position, csv_path in enumerate(csv_paths, 1):
            logger.run_header(position, len(csv_paths), csv_path)
            await process_csv(csv_path, who, None, overwrite)
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
            await process_csv(csv_path, who, model, overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICD-11 reverse lookup: code -> official title, then "
                    "consistency against the diagnosis.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo rows that already have icd11_lookup/consistency")
    args = parser.parse_args()

    csv_paths = find_run_csvs()
    if not csv_paths:
        print(f"[ERROR] No run{{1..{N_RUNS}}}.csv in {SCRIPT_DIR}")
        return
    print(f"[CSV] Iterative execution: {len(csv_paths)} runs found\n")

    who = WHOLookupClient(WHO_CLIENT_ID, WHO_CLIENT_SECRET)
    if not who.authenticate():
        return

    with RunLogger(LOG_PATH, enabled=LOGGING_ENABLED) as logger:
        logger.session_header({
            "phase": "reverse lookup + consistency",
            "icd version": "ICD-11",
            "model": "(skipped)" if SKIP_CONSISTENCY else MODEL_NAME,
            "lms host": LMS_HOST,
            "temperature": TEMPERATURE,
            "runs": len(csv_paths),
            "rows per run": MAX_ROWS if MAX_ROWS is not None else "all",
            "workers": WORKERS,
            "overwrite": args.overwrite,
            "eval NF rows": EVAL_NF,
            "who api": WHOLookupClient.BASE_URL,
            "titles language": WHOLookupClient.LANG,
        })
        asyncio.run(run_lookup(csv_paths, who, args.overwrite, logger))


if __name__ == "__main__":
    main()
