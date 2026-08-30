"""
icd-experiment.py (ICD-11) — assign ICD-11 codes to clinical diagnoses.

Cascade, entirely in ENGLISH over diagnosis_en:

    0. Disease         LLM   "which disease is this?" -> standard name
    1. Cache                 repeated diagnoses within a run
    2. Lexicon exact         literal hit in Orphanet
    3. Lexicon fuzzy   LLM   1 candidate, the LLM validates it
    4. WHO /search     LLM   wide pool, the LLM chooses one or none
    5. NF                    not found, sequential NF-XXXX

Steps 2 and 3 use the Orphanet lexical index, which covers rare diseases
only; anything common falls through to the WHO API in step 4. Both variants
— the complete phrase first, then the disease name from step 0 — are tried at
every step, and `match_source` records which one won.

Unlike the ICD-10 folder there is no offline catalogue and no exact-match
step against one: the WHO ICD-11 API returns ranked candidates rather than a
term index, so step 4 always goes through the LLM. That asymmetry follows
from what each classification publishes, not from a design choice.

Configuration lives in the CONFIG block below, not in command-line flags.

See MATCHING-METHODOLOGY.md for the full rationale, and icd_common.py for
everything shared with the ICD-10 pipeline.
"""

import asyncio
import json
import os
import re
from pathlib import Path

import lmstudio as lms
import requests
import urllib3
from tqdm import tqdm

import icd_common as common
from run_logger import RunLogger

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "medgemma-27b-it"
LMS_HOST = os.environ.get("LMS_HOST", "10.8.0.45:1234")

WHO_CLIENT_ID = os.environ.get(
    "WHO_CLIENT_ID",
    "00da56ea-0fe1-465e-b830-61cb0add2173_2741cb3a-0cd6-4977-af95-6258be8bd99a")
WHO_CLIENT_SECRET = os.environ.get(
    "WHO_CLIENT_SECRET", "h1MrlyMGBnGt7Q6kAnSpq8/1s18FkkzPboT7MIaim7o=")

# temperature=0 is greedy decoding: the argmax token is taken and the random
# generator is never consulted. Pinned here rather than inherited from the LM
# Studio UI, so runs stay comparable. A per-request seed does not exist in the
# SDK — `seed` applies at model load — and at temperature 0 it changes nothing
# either way. Any difference between runs is therefore NOT sampling variance
# but engine non-determinism (batching, float non-associativity, KV reuse).
TEMPERATURE = 0.0
SEED = 42
PREDICTION_CONFIG = {"temperature": TEMPERATURE}

FUZZY_THRESHOLD = 85       # step 3, strict: one candidate or nothing
SEARCH_LIMIT = 5           # candidates requested per WHO /search query
BATCH_SIZE = 20            # rows resolved concurrently
N_RUNS = 10
MAX_ROWS = None            # set to an int to process only the first N rows

PRODUCT1_PATH: str | None = None      # located automatically when None

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "experiment-runs.log"
LOGGING_ENABLED = True
NF_DICT_PATH = SCRIPT_DIR / "nf_dictionary.json"
CHAPTER_CACHE_PATH = SCRIPT_DIR / "chapter_cache.json"

ICD_VERSION = "11"
ICD_SOURCE = "ICD-11"
PRODUCT1_FILENAME = "en_product1.json"

COLUMNS = ["diagnosis_es", "diagnosis_en", "core_diagnosis", "clinical_summary",
           "icd11_code", "icd_code_complete", "category", "chapter",
           "hierarchical_distance", "hierarchy_path", "mapping_relation",
           "icd11_lookup", "match_type", "match_source"]

STAT_LABELS = [
    ("exact_complete", "Exact lexicon (complete)"),
    ("exact_core", "Exact lexicon (name)"),
    ("fuzzy_complete", "Fuzzy lexicon + LLM validates (complete)"),
    ("fuzzy_core", "Fuzzy lexicon + LLM validates (name)"),
    ("who_search", "WHO /search pool + LLM chooses"),
    ("nf", "NF (not found)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# WHO ICD-11 API
# ─────────────────────────────────────────────────────────────────────────────

class WHOClient:
    """
    WHO ICD-11 API client: free-text search plus hierarchy walking.

    Requests are always English — ICD-11 is authored in English and its
    translations index fewer terms, so results stay richer and comparable
    with the ICD-10 folder, which only supports English at all.
    """

    TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
    BASE_URL = "https://id.who.int/icd/release/11/2024-01/mms"
    SEARCH_URL = f"{BASE_URL}/search"
    CODEINFO_URL = f"{BASE_URL}/codeinfo"
    LANG = "en"

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None

    def authenticate(self) -> bool:
        if not self.client_id or not self.client_secret:
            print("[WHO] No credentials — WHO API disabled\n")
            return False
        try:
            response = requests.post(
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

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Accept-Language": self.LANG,
                "API-Version": "v2"}

    def _get_json(self, url: str, **kwargs):
        response = requests.get(url, headers=self._headers(), verify=False,
                                timeout=15, **kwargs)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def search(self, text: str, limit: int = SEARCH_LIMIT) -> list:
        """
        Candidates for a free-text diagnosis: [{icd11_code, title}, ...].

        No /autocode fallback: on the reference run that endpoint resolved
        nothing, and when the LLM rejects every /search candidate it tends to
        propose those same discarded codes.
        """
        if not self.token or not text.strip():
            return []
        try:
            data = self._get_json(
                self.SEARCH_URL,
                params={"q": text, "useFlexisearch": "true",
                        "flatResults": "true", "highlightingEnabled": "false",
                        "includeKeywordResult": "false"})
        except Exception as e:
            print(f"\n    [WHO-search] Error: {e}")
            return []

        candidates, seen = [], set()
        for entity in ((data or {}).get("destinationEntities") or []):
            code = (entity.get("theCode") or "").strip()
            # The API marks matched fragments with <em> tags.
            title = re.sub(r"</?em[^>]*>", "", entity.get("title", "")).strip()
            if code and title and code not in seen:
                seen.add(code)
                candidates.append({"icd11_code": code, "title": title})
                if len(candidates) >= limit:
                    break
        return candidates

    @staticmethod
    def _title(entity: dict) -> str:
        title = (entity or {}).get("title")
        return title.get("@value", "") if isinstance(title, dict) else (title or "")

    def get_hierarchy(self, code: str) -> tuple[str, str, int | None, str]:
        """
        Climb `parent` links to the chapter, returning
        (chapter_code, chapter_title, distance, path).

        The path is collected during the same walk — a second pass would pay
        the API twice — and tags each level with its classKind. ICD-11 blocks
        carry no `code`: their identifier is `codeRange` ("2B70-2C1Z"), the
        direct analogue of an ICD-10 block range, which keeps paths
        comparable between the two folders.

        Any API failure returns ("", "", None, "") and the caller degrades.
        """
        info = self._get_json(f"{self.CODEINFO_URL}/{code}")
        stem_id = (info or {}).get("stemId", "")
        current = self._get_json(stem_id) if stem_id else None
        if not current:
            return "", "", None, ""

        def step(entity: dict) -> str:
            label = ((entity.get("code") or "").strip()
                     or (entity.get("codeRange") or "").strip()
                     or (entity.get("blockId") or "").strip() or "?")
            return f"{label}[{(entity.get('classKind') or '?').strip()}]"

        chain, distance = [step(current)], 0
        while (current.get("classKind", "").lower() != "chapter"
               and current.get("parent") and distance < 10):
            parent = self._get_json(current["parent"][0])
            if not parent:
                break
            current = parent
            distance += 1
            chain.append(step(current))

        return ((current.get("code") or "").strip(), self._title(current),
                distance, " > ".join(reversed(chain)))


# ─────────────────────────────────────────────────────────────────────────────
# CHAPTER CACHE — deterministic hierarchy, safe to persist
# ─────────────────────────────────────────────────────────────────────────────

def load_chapter_cache() -> dict:
    if CHAPTER_CACHE_PATH.exists():
        return json.loads(CHAPTER_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_chapter_cache(cache: dict) -> None:
    CHAPTER_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


async def resolve_hierarchy(who: WHOClient, category: str,
                            cache: dict) -> tuple[str, str, str]:
    """
    (chapter, hierarchical_distance, hierarchy_path) for a category.

    Cached to disk because the classification hierarchy is deterministic:
    unlike the per-diagnosis caches this introduces no run-to-run coupling.
    Entries predating `hierarchy_path`, or written while blocks resolved to
    "?", are treated as stale and recomputed.
    """
    category = (category or "").strip()
    if not category:
        return "", "", ""
    if category.startswith("NF-"):
        return "NF", "0", ""      # not part of the classification

    entry = cache.get(category)
    if entry and "?[" not in entry.get("hierarchy_path", "?["):
        return (entry.get("chapter_title", ""),
                str(entry.get("hierarchical_distance", "")),
                entry.get("hierarchy_path", ""))

    chapter_code = chapter_title = path = ""
    distance = None
    if who.token:
        try:
            chapter_code, chapter_title, distance, path = await asyncio.to_thread(
                who.get_hierarchy, category)
        except Exception as e:
            print(f"\n    [WHO-chapter] {category}: {e}")

    cache[category] = {"chapter_code": chapter_code,
                       "chapter_title": chapter_title,
                       "hierarchical_distance": distance,
                       "hierarchy_path": path}
    return chapter_title, ("" if distance is None else str(distance)), path


# ─────────────────────────────────────────────────────────────────────────────
# CASCADE
# ─────────────────────────────────────────────────────────────────────────────

async def _who_pool(complete: str, core: str, model,
                    who: WHOClient) -> dict | None:
    """
    Step 4: query /search with BOTH variants, merge into one pool
    deduplicated by code, and let the LLM choose.

    The LLM always judges against the COMPLETE phrase, never the disease
    name, even when the candidate surfaced through the name — the cause is
    what decides the category (drug-induced diabetes rather than type 2).
    """
    if not who.token:
        return None

    queries = [("complete", complete)]
    if core and common.normalize(core) != common.normalize(complete):
        queries.append(("core", core))

    pool: dict = {}
    for source, query in queries:
        if not query:
            continue
        # requests is blocking: keep it off the event loop.
        for candidate in await asyncio.to_thread(who.search, query):
            pool.setdefault(candidate["icd11_code"],
                            {"title": candidate["title"], "source": source})
    if not pool:
        return None

    by_title: dict = {}
    for code, info in pool.items():
        by_title.setdefault(info["title"], (code, info["source"]))

    chosen = await common.semantic_match(complete, list(by_title), model,
                                         PREDICTION_CONFIG)
    if not chosen or chosen not in by_title:
        return None

    code, source = by_title[chosen]
    return {"input": complete, "matched_name": chosen, "orpha_code": None,
            "icd_codes": [{"code": code, "mapping_relation": ""}],
            "icd_code_complete": code, "mapping_relation": "",
            "match_type": "who_search", "match_source": source,
            "score": 0.0, "has_code": True}


async def resolve_diagnosis(diagnosis: str, lexicon: dict, model,
                            who: WHOClient, nf_dict: dict) -> dict:
    """
    Run the cascade for one diagnosis.

    The COMPLETE phrase is always tried before the disease name, being more
    specific; the name only widens recall. Semantic validation always happens
    against the complete phrase, so a misidentified disease produces a
    rejected candidate and an NF row — a visible failure rather than a silent
    wrong code.
    """
    if not diagnosis:
        return common.to_nf_result("", nf_dict)

    core = await common.identify_disease(diagnosis, model, PREDICTION_CONFIG)

    cache_key = common.normalize(diagnosis)
    if cache_key in common.lookup_cache:
        return common.lookup_cache[cache_key].copy()

    def finish(result: dict) -> dict:
        result.setdefault("match_source", "complete")
        result["core_diagnosis"] = core
        common.lookup_cache[cache_key] = result
        return result

    variants = [("complete", diagnosis)]
    if common.normalize(core) != cache_key:
        variants.append(("core", core))

    # 2. Lexicon exact — reliable, no validation needed
    for source, term in variants:
        result = common.lookup_exact(term, lexicon)
        if result and result["has_code"]:
            result = await common.finalize_node_result(
                diagnosis, result, model, PREDICTION_CONFIG, ICD_VERSION)
            result["match_type"] = f"exact_{source}"
            result["match_source"] = source
            return finish(result)

    # 3. Lexicon fuzzy — the LLM vetoes the single candidate
    for source, term in variants:
        result = common.lookup_fuzzy(term, lexicon, FUZZY_THRESHOLD)
        if result and result["has_code"]:
            confirmed = await common.semantic_match(
                diagnosis, [result["matched_name"]], model, PREDICTION_CONFIG)
            if confirmed:
                result = await common.finalize_node_result(
                    diagnosis, result, model, PREDICTION_CONFIG, ICD_VERSION)
                result["match_type"] = f"fuzzy_{source}"
                result["match_source"] = source
                return finish(result)

    # 4. WHO /search — the LLM chooses, or refuses
    if result := await _who_pool(diagnosis, core, model, who):
        return finish(result)

    # 5. NF
    result = common.to_nf_result(diagnosis, nf_dict)
    result["match_source"] = "none"
    return finish(result)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def process_row(row: dict, lexicon: dict, model, who: WHOClient,
                      nf_dict: dict, chapter_cache: dict) -> dict:
    """
    Fill one row's assignment columns. Only diagnosis_en feeds the pipeline;
    diagnosis_es is kept as source data but never read.
    """
    diagnosis = (row.get("diagnosis_en") or "").strip()
    if not diagnosis:
        row.update({c: "" for c in ("icd11_code", "icd_code_complete",
                                    "category", "chapter",
                                    "hierarchical_distance", "hierarchy_path",
                                    "mapping_relation", "core_diagnosis",
                                    "match_type", "match_source")})
        return row

    result = await resolve_diagnosis(diagnosis, lexicon, model, who, nf_dict)
    complete_code = result.get("icd_code_complete", "")
    category = common.truncate_code(complete_code)
    chapter, distance, path = await resolve_hierarchy(who, category, chapter_cache)

    row.update({
        "icd11_code": category,          # backwards compatible: equals category
        "icd_code_complete": complete_code,
        "category": category,
        "chapter": chapter,
        "hierarchical_distance": distance,
        "hierarchy_path": path,
        "mapping_relation": result.get("mapping_relation", ""),
        "core_diagnosis": result.get("core_diagnosis", ""),
        "match_type": result["match_type"],
        "match_source": result.get("match_source", ""),
    })
    return row


async def process_csv(csv_path: str, lexicon: dict, model, who: WHOClient,
                      nf_dict: dict, chapter_cache: dict) -> None:
    """Process one run's CSV in place, reusing the loaded index and model."""
    rows, fieldnames = common.read_csv_robust(csv_path, COLUMNS)
    if not rows:
        print(f"[CSV] {csv_path} has no data rows.\n")
        return

    pending = rows if MAX_ROWS is None else rows[:MAX_ROWS]
    with_diagnosis = sum(1 for r in pending
                         if (r.get("diagnosis_en") or "").strip())
    print(f"[CSV] {len(rows)} rows loaded | diagnosis_en: {with_diagnosis}")
    if not with_diagnosis:
        print("[!] No row has diagnosis_en: everything will fall to NF.")
    print()

    with tqdm(total=len(pending), desc=f"Processing {Path(csv_path).name}",
              unit="row") as bar:
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            await asyncio.gather(*(
                process_row(row, lexicon, model, who, nf_dict, chapter_cache)
                for row in batch))
            bar.update(len(batch))

    common.print_match_stats(pending, STAT_LABELS, ICD_SOURCE)
    common.write_csv(csv_path, rows, fieldnames)


def find_run_csvs() -> list[str]:
    """run1.csv .. runN.csv sitting next to the script."""
    paths = [SCRIPT_DIR / f"run{i}.csv" for i in range(1, N_RUNS + 1)]
    if missing := [p.name for p in paths if not p.exists()]:
        print(f"[CSV] Notice: not found {missing}")
    return [str(p) for p in paths if p.exists()]


async def run_experiment(product1_path: str, csv_paths: list[str],
                         logger: RunLogger) -> None:
    lexicon = common.build_lexical_index(product1_path, ICD_SOURCE)
    who = WHOClient(WHO_CLIENT_ID, WHO_CLIENT_SECRET)
    who.authenticate()
    nf_dict = common.load_nf_dictionary(NF_DICT_PATH)
    nf_before = len(nf_dict)
    print(f"[NF]  Dictionary loaded: {nf_before} entries\n")

    chapter_cache = load_chapter_cache()
    print(f"[Chapters] Cache loaded: {len(chapter_cache)} categories\n")

    logger.session_header({
        "icd version": ICD_SOURCE,
        "model": MODEL_NAME,
        "lms host": LMS_HOST,
        "temperature": TEMPERATURE,
        "seed (load-time)": SEED,
        "runs": len(csv_paths),
        "fuzzy threshold": FUZZY_THRESHOLD,
        "rows per run": MAX_ROWS if MAX_ROWS is not None else "all",
        "who api": "authenticated" if who.token else "UNAVAILABLE",
        "product1": product1_path,
        "lexicon": f"{len(lexicon)} entries",
    })

    async with lms.AsyncClient(LMS_HOST) as client:
        print(f"[LLM] Host: {LMS_HOST}\n[LLM] Connecting to: {MODEL_NAME}")
        try:
            # The seed only applies if this call actually loads the model; it
            # is ignored when LM Studio already holds it, and is moot at
            # temperature 0 anyway.
            model = await client.llm.model(MODEL_NAME, config={"seed": SEED})
        except Exception as e:
            print(f"[LLM] ERROR connecting to {LMS_HOST}: {e}")
            print("[LLM] Check LM Studio is serving on that IP:port "
                  "(Settings > Developer > Serve on Local Network).\n")
            return
        print("[LLM] Connected\n")

        for index, csv_path in enumerate(csv_paths, 1):
            logger.run_header(index, len(csv_paths), csv_path)
            # Cold start per run: without this, runs 2..10 are served from
            # run 1's cache and the experiment measures nothing.
            common.reset_run_caches()
            await process_csv(csv_path, lexicon, model, who, nf_dict,
                              chapter_cache)

    common.save_nf_dictionary(nf_dict, NF_DICT_PATH)
    save_chapter_cache(chapter_cache)
    print(f"[NF]  {nf_before} -> {len(nf_dict)} entries "
          f"(+{len(nf_dict) - nf_before}), saved to {NF_DICT_PATH}")
    print(f"[Chapters] Cache saved: {len(chapter_cache)} categories\n")


def main() -> None:
    product1_path = common.find_data_file(PRODUCT1_FILENAME, PRODUCT1_PATH)
    if not product1_path:
        print(f"[ERROR] {PRODUCT1_FILENAME} not found near {SCRIPT_DIR}. "
              f"Copy it next to the script or set PRODUCT1_PATH.")
        return
    print(f"[Lexical index] product1 detected: {product1_path}\n")

    csv_paths = find_run_csvs()
    if not csv_paths:
        print(f"[ERROR] No run{{1..{N_RUNS}}}.csv in {SCRIPT_DIR}")
        return
    print(f"[CSV] Iterative execution: {len(csv_paths)} runs found\n")

    with RunLogger(LOG_PATH, enabled=LOGGING_ENABLED) as logger:
        asyncio.run(run_experiment(product1_path, csv_paths, logger))


if __name__ == "__main__":
    main()
