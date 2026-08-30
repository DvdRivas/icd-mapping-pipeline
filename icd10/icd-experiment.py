"""
icd-experiment.py (ICD-10) — assign ICD-10 codes to clinical diagnoses.

Cascade, entirely in ENGLISH over diagnosis_en:

    0.  Disease         LLM   "which disease is this?" -> standard name
    1.  Cache                 repeated diagnoses within a run
    2.  Lexicon exact         literal hit in Orphanet
    3.  Lexicon fuzzy   LLM   1 candidate, the LLM validates it
    4a. Catalogue exact       literal official term or synonym, no LLM
    4b. Catalogue pool  LLM   wide pool, the LLM chooses one or none
    5.  NF                    not found, sequential NF-XXXX

Steps 2 and 3 use the Orphanet lexical index, which covers rare diseases
only; anything common falls through to the ClaML catalogue in steps 4a/4b.
Both variants — the complete phrase first, then the disease name from step 0
— are tried at every step, and `match_source` records which one won.

Configuration lives in the CONFIG block below, not in command-line flags.

See MATCHING-METHODOLOGY.md for the full rationale, and icd_common.py for
everything shared with the ICD-11 pipeline.
"""

import asyncio
import os
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
CATALOGUE_POOL = 10        # step 4b, wide: the LLM filters
BATCH_SIZE = 20            # rows resolved concurrently
N_RUNS = 10
MAX_ROWS = None            # set to an int to process only the first N rows

# Data files are located automatically; set a path to override.
PRODUCT1_PATH: str | None = None
CLAML_PATH: str | None = None

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "experiment-runs.log"
LOGGING_ENABLED = True
NF_DICT_PATH = SCRIPT_DIR / "nf_dictionary.json"

ICD_VERSION = "10"
ICD_SOURCE = "ICD-10"
PRODUCT1_FILENAME = "en_product1.json"

COLUMNS = ["diagnosis_es", "diagnosis_en", "core_diagnosis", "clinical_summary",
           "icd10_code", "icd_code_complete", "category", "chapter",
           "hierarchical_distance", "hierarchy_path", "mapping_relation",
           "icd10_lookup", "match_type", "match_source"]

STAT_LABELS = [
    ("exact_complete", "Exact lexicon (complete)"),
    ("exact_core", "Exact lexicon (name)"),
    ("fuzzy_complete", "Fuzzy lexicon + LLM validates (complete)"),
    ("fuzzy_core", "Fuzzy lexicon + LLM validates (name)"),
    ("catalog_exact_complete", "Catalogue exact, no LLM (complete)"),
    ("catalog_exact_core", "Catalogue exact, no LLM (name)"),
    ("catalog_search", "Catalogue pool + LLM chooses"),
    ("nf", "NF (not found)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────

class ICD10Catalog:
    """
    Stands in for the "WHO /search" step of the ICD-11 pipeline.

    The WHO ICD-10 API exposes neither /search nor /codeinfo (both 404) and
    answers in English only, so the catalogue is parsed from the official
    ClaML XML instead — offline, in under a second, no credentials.

    Beyond replacing that API, the XML supplies 9453 `inclusion` rubrics (the
    official synonyms), which is what makes an EXACT catalogue match viable:
    roughly a third of the dataset resolves literally, with no LLM call.

    Without the XML `available` stays False and both catalogue steps are
    skipped: rows fall to NF instead of breaking.
    """

    def __init__(self, index: ICD10Index | None):
        self.index = index

    @property
    def available(self) -> bool:
        return self.index is not None and len(self.index) > 0

    @classmethod
    def load(cls, claml_path: str | None = None) -> "ICD10Catalog":
        index = ICD10Index.load(claml_path)
        if index is None:
            print(f"[ICD-10] {CLAML_FILENAME} not found: catalogue steps stay "
                  f"idle and uncovered diagnoses will fall to NF.\n")
        else:
            print(f"[ICD-10] Catalogue: {len(index)} classes | "
                  f"{len(index.terms)} searchable terms | version {index.version}")
            print(f"[ICD-10] Source: {index.source}\n")
        return cls(index)

    def lookup_exact(self, text: str) -> str | None:
        """Literal match on an official term or synonym. Deterministic."""
        if not self.available or not (text or "").strip():
            return None
        code = self.index.lookup_exact(text)
        # Only categories are assignable; blocks exist for the hierarchy.
        return code if code and self.index.is_category(code) else None

    def search(self, text: str, limit: int = CATALOGUE_POOL) -> list:
        """
        Fuzzy candidate pool. Deliberately wide: string similarity cannot
        settle ICD-10 wording, so recall is generous and the LLM discards.
        """
        if not self.available or not text.strip():
            return []
        return [{"icd10_code": r["icd10_code"], "title": r["title"]}
                for r in self.index.search(text, limit=limit)
                if self.index.is_category(r["icd10_code"])]


async def resolve_hierarchy(catalog: ICD10Catalog, category: str) -> tuple[str, str, str]:
    """
    (chapter, hierarchical_distance, hierarchy_path) for a category.

    All three are measured by walking the ClaML tree, never inferred from the
    code string: blocks nest irregularly, so C22 sits 4 levels below its
    chapter while E84 sits 2. The path makes the distance auditable — its
    element count is always distance + 1.
    """
    category = (category or "").strip()
    if not category:
        return "", "", ""
    if category.startswith("NF-"):
        return "NF", "0", ""      # not part of the classification

    if catalog.available and category in catalog.index:
        distance = catalog.index.hierarchical_distance(category)
        return (catalog.index.chapter(category),
                "" if distance is None else str(distance),
                catalog.index.hierarchy_path(category))
    return "", "", ""


# ─────────────────────────────────────────────────────────────────────────────
# CASCADE
# ─────────────────────────────────────────────────────────────────────────────

async def _catalogue_pool(complete: str, core: str, model,
                          catalog: ICD10Catalog) -> dict | None:
    """
    Step 4b: search with BOTH variants, merge into one pool deduplicated by
    code, and let the LLM choose.

    The LLM always judges against the COMPLETE phrase, never the disease name,
    even when the candidate surfaced through the name — the cause is what
    decides. A pyogenic abscess must not land on the amoebic code just because
    both mention "liver abscess".
    """
    if not catalog.available:
        return None

    queries = [("complete", complete)]
    if core and common.normalize(core) != common.normalize(complete):
        queries.append(("core", core))

    pool: dict = {}
    for source, query in queries:
        if not query:
            continue
        for candidate in await asyncio.to_thread(catalog.search, query):
            pool.setdefault(candidate["icd10_code"],
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
            "match_type": "catalog_search", "match_source": source,
            "score": 0.0, "has_code": True}


async def resolve_diagnosis(diagnosis: str, lexicon: dict, model,
                            catalog: ICD10Catalog, nf_dict: dict) -> dict:
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

    # 4a. Catalogue exact — as reliable as step 2, and free of the LLM
    for source, term in variants:
        if code := catalog.lookup_exact(term):
            return finish({"input": diagnosis,
                           "matched_name": catalog.index.title(code),
                           "orpha_code": None,
                           "icd_codes": [{"code": code, "mapping_relation": ""}],
                           "icd_code_complete": code, "mapping_relation": "",
                           "match_type": f"catalog_exact_{source}",
                           "match_source": source, "score": 100.0,
                           "has_code": True})

    # 4b. Catalogue pool — the LLM chooses, or refuses
    if result := await _catalogue_pool(diagnosis, core, model, catalog):
        return finish(result)

    # 5. NF
    result = common.to_nf_result(diagnosis, nf_dict)
    result["match_source"] = "none"
    return finish(result)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

async def process_row(row: dict, lexicon: dict, model, catalog: ICD10Catalog,
                      nf_dict: dict) -> dict:
    """
    Fill one row's assignment columns. Only diagnosis_en feeds the pipeline;
    diagnosis_es is kept as source data but never read.
    """
    diagnosis = (row.get("diagnosis_en") or "").strip()
    if not diagnosis:
        row.update({c: "" for c in ("icd10_code", "icd_code_complete",
                                    "category", "chapter",
                                    "hierarchical_distance", "hierarchy_path",
                                    "mapping_relation", "core_diagnosis",
                                    "match_type", "match_source")})
        return row

    result = await resolve_diagnosis(diagnosis, lexicon, model, catalog, nf_dict)
    complete_code = result.get("icd_code_complete", "")
    category = common.truncate_code(complete_code)
    chapter, distance, path = await resolve_hierarchy(catalog, category)

    row.update({
        "icd10_code": category,          # backwards compatible: equals category
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


async def process_csv(csv_path: str, lexicon: dict, model,
                      catalog: ICD10Catalog, nf_dict: dict) -> None:
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
                process_row(row, lexicon, model, catalog, nf_dict)
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
    catalog = ICD10Catalog.load(CLAML_PATH)
    nf_dict = common.load_nf_dictionary(NF_DICT_PATH)
    nf_before = len(nf_dict)
    print(f"[NF]  Dictionary loaded: {nf_before} entries\n")

    logger.session_header({
        "icd version": ICD_SOURCE,
        "model": MODEL_NAME,
        "lms host": LMS_HOST,
        "temperature": TEMPERATURE,
        "seed (load-time)": SEED,
        "runs": len(csv_paths),
        "fuzzy threshold": FUZZY_THRESHOLD,
        "rows per run": MAX_ROWS if MAX_ROWS is not None else "all",
        "catalogue": (f"{len(catalog.index)} classes, "
                      f"version {catalog.index.version}"
                      if catalog.available else "UNAVAILABLE"),
        "catalogue source": catalog.index.source if catalog.available else "-",
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
            await process_csv(csv_path, lexicon, model, catalog, nf_dict)

    common.save_nf_dictionary(nf_dict, NF_DICT_PATH)
    print(f"[NF]  {nf_before} -> {len(nf_dict)} entries "
          f"(+{len(nf_dict) - nf_before}), saved to {NF_DICT_PATH}\n")


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
