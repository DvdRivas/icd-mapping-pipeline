"""
icd_common.py — pieces shared by the ICD-10 and ICD-11 pipelines.

Identical copy in both folders, like deep_analysis.py and run_logger.py: each
folder stays self-contained and runnable on its own, while the shared logic
lives in one conceptual place. Several bugs in this project had to be fixed
twice because the two pipelines drifted; anything version-agnostic belongs
here so that stops happening.

What is NOT here: the reference sources. ICD-10 reads an offline ClaML XML
and ICD-11 queries the WHO API, because that is what each classification
publishes. Those live in their own modules.
"""

import csv
import json
import re
from datetime import date
from pathlib import Path

import lmstudio as lms
from rapidfuzz import fuzz, process

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_SEARCH_DEPTH = 3          # parent directories walked when locating data


# ─────────────────────────────────────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

_APOSTROPHES_RE = re.compile(r"[‘’´`']")
_BRACKETS_RE = re.compile(r"[\[\]()]")
_DAGGER_RE = re.compile(r"[+*†]+$")


def normalize(text: str) -> str:
    """
    Single normalization for every string comparison in both pipelines.

    Each rule fixes an observed failure:
      - apostrophes: "Parkinson's" with a curly quote never matched the same
        string with a straight one, nor "Parkinsons".
      - brackets: classifications write eponyms as "Aortic arch syndrome
        [Takayasu]". Glued to the word, "[takayasu]" never matches the token
        "takayasu", which made those entities unreachable by name.
    """
    text = (text or "").lower().strip()
    text = _APOSTROPHES_RE.sub("", text)
    text = _BRACKETS_RE.sub(" ", text)
    text = re.sub(r"[\-–—]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_dagger(code: str) -> str:
    """
    Drop the dagger/asterisk marker of dual classification.

    Orphanet writes some references as "A39.1+" or "B00.4*" while the
    classification stores them unmarked, so without stripping they fail to
    match silently.
    """
    return _DAGGER_RE.sub("", (code or "").strip())


def truncate_code(code: str) -> str:
    """
    Category of a code: everything left of the first dot.
    "5A11.2" -> "5A11", "A49.1" -> "A49". NF codes are left untouched.
    """
    code = (code or "").strip()
    return code if not code or code.startswith("NF-") else code.split(".", 1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# FILE LOCATION
# ─────────────────────────────────────────────────────────────────────────────

def find_data_file(filename: str, explicit: str | None = None,
                   base_dir: Path = SCRIPT_DIR) -> str | None:
    """
    Locate a data file without depending on the working directory.

    Walks from the script folder upwards, checking each level and its direct
    subfolders, so a local copy wins over a shared one in a sibling folder.
    """
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.is_file() else None

    for directory in [base_dir, *list(base_dir.parents)[:DATA_SEARCH_DEPTH]]:
        if (direct := directory / filename).is_file():
            return str(direct)
        for match in sorted(directory.glob(f"*/{filename}")):
            return str(match)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LEXICAL INDEX (Orphanet product1)
# ─────────────────────────────────────────────────────────────────────────────

def build_lexical_index(product1_path: str, icd_source: str) -> dict:
    """
    Map every normalized name and synonym to its Orphanet disorder node.

    Not a knowledge graph: no relations between nodes, only a name -> code
    lookup table, hence "lexical index". `icd_source` selects which external
    references to keep ("ICD-10" or "ICD-11").

    Node: {canonical_name, orpha_code, icd_codes: [{code, mapping_relation}]}
    """
    print(f"[Lexical index] Loading {product1_path} ...")
    with open(product1_path, encoding="utf-8") as f:
        data = json.load(f)

    disorders = (data.get("JDBOR", [{}])[0]
                     .get("DisorderList", [{}])[0]
                     .get("Disorder", []))

    index, with_code = {}, 0
    for disorder in disorders:
        names = disorder.get("Name", [])
        canonical = _pick_label(names)
        if not canonical:
            continue

        codes = _extract_icd_codes(disorder, icd_source)
        with_code += bool(codes)
        node = {"canonical_name": canonical,
                "orpha_code": disorder.get("OrphaCode", ""),
                "icd_codes": codes}

        index[normalize(canonical)] = node          # canonical wins
        for block in disorder.get("SynonymList", []):
            for synonym in block.get("Synonym", []):
                if key := normalize(synonym.get("label", "")):
                    index.setdefault(key, node)     # synonyms never overwrite

    print(f"[Lexical index] {len(disorders)} disorders | {with_code} with "
          f"{icd_source} | {len(index)} total entries\n")
    return index


def _pick_label(names: list, lang: str = "en") -> str:
    for name in names:
        if name.get("lang") == lang:
            return name["label"].strip()
    return names[0]["label"].strip() if names else ""


def _extract_icd_codes(disorder: dict, icd_source: str) -> list:
    """Codes from ExternalReferenceList, each with its mapping relation."""
    codes = []
    for block in disorder.get("ExternalReferenceList", []):
        for ref in block.get("ExternalReference", []):
            if ref.get("Source") != icd_source:
                continue
            if code := strip_dagger(ref.get("Reference", "")):
                codes.append({"code": code,
                              "mapping_relation": _relation_code(ref)})
    return codes


def _relation_code(ref: dict) -> str:
    """Short form (E, NTBT, BTNT, ND) of DisorderMappingRelation."""
    relations = ref.get("DisorderMappingRelation") or []
    if not relations:
        return "ND"
    label = relations[0].get("Name", [{}])[0].get("label", "")
    return label.split()[0].strip() if label else "ND"


def make_node_result(diagnosis: str, node: dict, match_type: str,
                     score: float) -> dict:
    return {"input": diagnosis,
            "matched_name": node["canonical_name"],
            "orpha_code": node["orpha_code"],
            "icd_codes": node["icd_codes"],
            "match_type": match_type,
            "score": score,
            "has_code": bool(node["icd_codes"])}


def lookup_exact(diagnosis: str, index: dict) -> dict | None:
    node = index.get(normalize(diagnosis))
    return make_node_result(diagnosis, node, "exact", 100.0) if node else None


def lookup_fuzzy(diagnosis: str, index: dict, threshold: int) -> dict | None:
    """Single best fuzzy candidate; the caller has the LLM validate it."""
    result = process.extractOne(normalize(diagnosis), list(index),
                                scorer=fuzz.token_sort_ratio,
                                score_cutoff=threshold)
    if not result:
        return None
    key, score, _ = result
    return make_node_result(diagnosis, index[key], "fuzzy", round(score, 1))


# ─────────────────────────────────────────────────────────────────────────────
# NF DICTIONARY — identifiers for diagnoses no source could match
# ─────────────────────────────────────────────────────────────────────────────

def load_nf_dictionary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_nf_dictionary(nf_dict: dict, path: Path) -> None:
    path.write_text(json.dumps(nf_dict, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def get_or_create_nf_code(diagnosis: str, nf_dict: dict) -> str:
    """
    Existing NF code for the diagnosis, or a new sequential one.

    The dictionary only NAMES unmatched cases — it is written after the
    cascade already failed and never influences a decision — so it persists
    across runs to keep NF codes comparable.
    """
    key = normalize(diagnosis)
    if key in nf_dict:
        return nf_dict[key]["code"]

    highest = max((int(m.group(1))
                   for entry in nf_dict.values()
                   if (m := re.match(r"NF-(\d+)$", entry.get("code", "")))),
                  default=0)
    code = f"NF-{highest + 1:04d}"
    nf_dict[key] = {"code": code, "label": diagnosis,
                    "first_seen": date.today().isoformat()}
    return code


def to_nf_result(diagnosis: str, nf_dict: dict) -> dict:
    code = get_or_create_nf_code(diagnosis, nf_dict)
    return {"input": diagnosis, "matched_name": None, "orpha_code": None,
            "icd_codes": [{"code": code, "mapping_relation": ""}],
            "icd_code_complete": code, "mapping_relation": "",
            "match_type": "nf", "score": 0.0, "has_code": True}


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_IDENTIFICATION_PROMPT = """You are a clinician identifying diseases for medical classification.

You are given a clinical diagnosis as written in a case report. It usually
carries the causative agent, the clinical context and the comorbidities along
with the disease itself.

Answer one question: WHICH DISEASE IS THIS?

Reply with the standard name of that disease — the name under which it would
be listed in a medical classification such as ICD. Use your medical knowledge
to name the entity; do not merely delete words from the input. The standard
name often uses vocabulary that never appears in the input.

Requirements for your answer:
- It must be a NAMED DISEASE, never a bare symptom, sign or body part.
- It must be the disease actually present in the patient, not the trigger,
  the comorbidity or the setting.
- Prefer the most specific disease name that is a real classification entry.
  If the highly specific variant is not a classified entity, name the disease
  it belongs to.
- If the input is already the standard disease name, repeat it unchanged.

Answer with the disease name ONLY. No quotes, no explanation, no preamble.

════════ EXAMPLES ════════
Input:  Acute intermittent porphyria due to HMBS mutation
Answer: Acute intermittent porphyria

Input:  Cryptococcus neoformans meningoencephalitis
Answer: Cryptococcosis

Input:  Amoxicillin-induced rash in Epstein-Barr virus infectious mononucleosis
Answer: Drug eruption

Input:  Minocycline-induced hyperpigmentation type III
Answer: Drug-induced hyperpigmentation

Input:  Immune checkpoint inhibitor (pembrolizumab)-induced diabetes mellitus in metastatic melanoma
Answer: Drug-induced diabetes mellitus

Input:  Pediatric Takayasu arteritis
Answer: Takayasu arteritis

Input:  Vascular Ehlers-Danlos syndrome due to COL3A1 mutation
Answer: Vascular Ehlers-Danlos syndrome

Input:  Disseminated bartonellosis due to Bartonella henselae
Answer: Bartonellosis

Input:  Generalized argyria of unidentified origin
Answer: Argyria

Input:  Post-traumatic Morel-Lavallee lesion
Answer: Morel-Lavallee lesion

Input:  Emphysematous vertebral osteomyelitis due to ESBL-producing Klebsiella pneumoniae
Answer: Vertebral osteomyelitis

Input:  Free-floating iris pigment epithelial cyst in the vitreous
Answer: Iris cyst

Input:  Spur cell hemolytic anemia in advanced alcoholic cirrhosis
Answer: Acquired haemolytic anaemia

Input:  Jejunal variceal bleeding secondary to noncirrhotic portal hypertension
Answer: Bleeding intestinal varices

Input:  Ectopic intranasal tooth
Answer: Ectopic tooth

Input:  Pearly penile papules (normal anatomical variant)
Answer: Pearly penile papules

Input:  Secondary hemophagocytic lymphohistiocytosis due to COVID-19
Answer: Secondary haemophagocytic lymphohistiocytosis

Input:  Merkel cell carcinoma
Answer: Merkel cell carcinoma"""


SEMANTIC_MATCH_PROMPT = """You are a specialist in medical terminology.
Your task is to determine whether any of the candidate terms from a medical
ontology is a clinical equivalent of the input diagnosis.

Equivalence rules:
- Accept general clinical equivalences: "Pulmonary tuberculosis" is equivalent to
  "Primary pulmonary tuberculosis" because in a clinical setting they refer to
  the same disease process.
- Accept when the candidate is a more specific form of the same concept.
- Reject (null) when the candidate is a clearly different concept, even if it
  shares words with the input.
- If several candidates are valid, choose the clinically most precise one.
- Answer ONLY with the exact candidate term, or null if none is equivalent.
- No quotes, no explanations, just the term or null.

Examples:
  Input: "Pulmonary tuberculosis"
  Candidates: ["Primary pulmonary tuberculosis", "Miliary tuberculosis", "Tuberculosis"]
  Answer: Primary pulmonary tuberculosis

  Input: "Idiopathic hemoptysis"
  Candidates: ["Idiopathic pulmonary hemosiderosis", "Neonatal pulmonary hemorrhage"]
  Answer: null

  Input: "Nontuberculous mycobacteria"
  Candidates: ["Pulmonary nontuberculous mycobacterial infection", "Multifocal tuberculosis"]
  Answer: Pulmonary nontuberculous mycobacterial infection"""


def disambiguation_prompt(icd_version: str) -> str:
    return f"""You are a clinical coder choosing the single best ICD-{icd_version} code
for a diagnosed disease, when Orphanet maps that disease to more than one
ICD-{icd_version} code.

Each candidate code carries a mapping relation:
  E    - Exact mapping: the disease and the code are clinically equivalent.
  NTBT - The disease is NARROWER than the code (the code is broader/more
         general than the disease).
  BTNT - The disease is BROADER than the code (the code is a more specific
         subtype of the disease).
  ND   - The relation has not been decided.

Given the disease name and the full diagnosis (which may specify a subtype,
cause or context), pick the candidate code whose scope best matches the
diagnosis:
  - Prefer "E" when it is clinically adequate for the diagnosis.
  - If the diagnosis text points at a specific subtype and one candidate
    (often "BTNT") targets exactly that subtype, prefer it over a broader
    "E"/"NTBT" candidate.
  - If nothing distinguishes the candidates clinically, prefer "E", then the
    first candidate listed.

Answer ONLY with the exact code string of the chosen candidate. No quotes,
no explanation, no preamble."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM PLUMBING
# ─────────────────────────────────────────────────────────────────────────────

def parse_channel_response(raw: str) -> str:
    """Extract the final channel when the model uses the channel format."""
    if match := re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|\Z)",
                          raw, re.DOTALL):
        return match.group(1).strip()
    if "<|channel|>" in raw:
        last = raw.split("<|end|>")[-1].strip()
        if last and (cleaned := re.sub(r"<\|[^|]+\|>", "", last).strip()):
            return cleaned
    return raw.strip()


def clean_response(raw: str) -> str:
    """Strip markdown fences and surrounding quotes."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip().strip('"').strip("'")


async def _ask(model, system_prompt: str, user_message: str, config: dict) -> str:
    chat = lms.Chat(system_prompt)
    chat.add_user_message(user_message)
    result = await model.respond(chat, config=config)
    return clean_response(parse_channel_response(result.content))


# ─────────────────────────────────────────────────────────────────────────────
# LLM STEPS
# ─────────────────────────────────────────────────────────────────────────────

# Signs the model returned prose instead of a disease name.
_PROSE_RE = re.compile(r"[.;:]\s|\b(the|this|is|are|refers|means|should)\b",
                       re.IGNORECASE)

disease_cache: dict = {}
lookup_cache: dict = {}


def reset_run_caches() -> None:
    """
    Clear the per-diagnosis caches so every run starts cold.

    CRITICAL. The 10 runs share one process and all 10 CSVs hold the SAME
    diagnoses, so without this reset run 1 fills `lookup_cache` and runs 2..10
    are served entirely from it: no LLM call, byte-identical outputs, and
    deep_analysis.py reporting perfect stability that is purely an artefact.

    Within a run both caches are wanted — they stop repeated diagnoses from
    being recomputed. Only their survival ACROSS runs breaks the measurement.

    Deliberately NOT reset: the catalogue, the lexical index and the chapter
    cache (deterministic reference data), and nf_dictionary.json (identifiers
    that must stay comparable across runs).
    """
    disease_cache.clear()
    lookup_cache.clear()


def is_valid_disease_name(raw: str) -> bool:
    """
    Shape check, not a length check: naming the entity can yield a longer
    phrase than the input ("Jejunal variceal bleeding" -> "Bleeding
    intestinal varices"). Between 1 and 10 words, no sentence punctuation.
    """
    raw = (raw or "").strip()
    return (len(raw) >= 3 and 1 <= len(raw.split()) <= 10
            and not _PROSE_RE.search(raw))


async def identify_disease(diagnosis: str, model, config: dict) -> str:
    """
    Step 0: ask which disease the diagnosis is, and return its standard name.

    Clinical identification, not text trimming: the answer may use vocabulary
    absent from the input ("Cryptococcus neoformans meningoencephalitis" ->
    "Cryptococcosis"). No textual heuristic can decide in advance which rows
    need it, so it runs for every row — which also makes it the dominant
    source of run-to-run variance.

    On error the original diagnosis is returned, degrading instead of failing.
    """
    diagnosis = (diagnosis or "").strip()
    if not diagnosis:
        return diagnosis

    key = normalize(diagnosis)
    if key in disease_cache:
        return disease_cache[key]

    name = diagnosis
    try:
        raw = await _ask(model, DISEASE_IDENTIFICATION_PROMPT,
                         f"Input:  {diagnosis}\nAnswer:", config)
        raw = re.sub(r"^Answer:\s*", "", raw.split("\n")[0].strip(),
                     flags=re.IGNORECASE).strip()
        if is_valid_disease_name(raw):
            name = raw
        elif raw:
            print(f"\n    [LLM-disease] Answer discarded: {raw[:80]!r}")
    except Exception as e:
        print(f"\n    [LLM-disease] Error: {e}")

    disease_cache[key] = name
    return name


async def semantic_match(diagnosis: str, candidates: list, model,
                         config: dict) -> str | None:
    """
    The LLM picks the equivalent candidate, or None when none is.

    Called from two steps, and the candidate count changes what is asked:
      - fuzzy lexicon step, ONE candidate: string similarity already decided
        and the LLM acts as a VETO, letting the row continue when it rejects.
      - catalogue/search step, a POOL: similarity cannot settle the wording of
        a classification, so recall is wide and the LLM SELECTS. Answering
        null is valid and sends the row to NF.

    Either way it can return None, which is what stops a bad retrieval from
    becoming a wrong code.
    """
    if not candidates:
        return None

    raw = ""
    try:
        raw = await _ask(
            model, SEMANTIC_MATCH_PROMPT,
            f'Input diagnosis: "{diagnosis}"\n'
            f'Candidates: {json.dumps(candidates, ensure_ascii=False)}\n'
            f'Which candidate is equivalent? Answer only the exact term or null.',
            config)
        if raw.lower() == "null" or not raw:
            return None

        for candidate in candidates:
            if raw.lower() == candidate.lower():
                return candidate

        # Tolerate minor wording drift in the answer
        best = process.extractOne(raw.lower(), [c.lower() for c in candidates],
                                  scorer=fuzz.ratio, score_cutoff=90)
        return candidates[best[2]] if best else None
    except Exception as e:
        print(f"\n    [LLM-semantic] Error: {e} | raw: {raw[:200]}")
        return None


async def disambiguate_icd_code(diagnosis: str, disease_name: str,
                                candidates: list, model, config: dict,
                                icd_version: str) -> tuple[str, str]:
    """
    Pick one code among several mapped to the same Orphanet disorder, using
    DisorderMappingRelation plus the FULL diagnosis as clinical context — the
    cause may point at a subtype the disorder name alone does not.

    A single candidate returns immediately without calling the LLM (~96% of
    disorders). On error the tie-break is deterministic: prefer "E", else the
    first candidate.
    """
    seen: dict[str, str] = {}
    for candidate in candidates:
        if (code := candidate.get("code", "")) and code not in seen:
            seen[code] = candidate.get("mapping_relation", "ND")
    candidates = [{"code": c, "mapping_relation": r} for c, r in seen.items()]

    if not candidates:
        return "", ""
    if len(candidates) == 1:
        return candidates[0]["code"], candidates[0]["mapping_relation"]

    def fallback() -> tuple[str, str]:
        for candidate in candidates:
            if candidate["mapping_relation"] == "E":
                return candidate["code"], candidate["mapping_relation"]
        return candidates[0]["code"], candidates[0]["mapping_relation"]

    listing = "\n".join(f'- code: "{c["code"]}", '
                        f'mapping_relation: "{c["mapping_relation"]}"'
                        for c in candidates)
    raw = ""
    try:
        raw = await _ask(
            model, disambiguation_prompt(icd_version),
            f'Disease name: "{disease_name}"\nDiagnosis: "{diagnosis}"\n'
            f'Candidates:\n{listing}\n\n'
            f'Which code is the best match? Answer only the code.',
            config)
        normalized = re.sub(r"\s+", "", raw).upper()
        for candidate in candidates:
            if raw == candidate["code"] or normalized == re.sub(
                    r"\s+", "", candidate["code"]).upper():
                return candidate["code"], candidate["mapping_relation"]
        print(f"\n    [LLM-disambig] Unrecognized code: {raw[:60]!r} — "
              f"applying deterministic tie-break")
    except Exception as e:
        print(f"\n    [LLM-disambig] Error: {e} | raw: {raw[:100]}")

    return fallback()


async def finalize_node_result(diagnosis: str, result: dict, model,
                               config: dict, icd_version: str) -> dict:
    """Reduce a node's `icd_codes` to the single code that will be reported."""
    code, relation = await disambiguate_icd_code(
        diagnosis, result["matched_name"], result["icd_codes"], model,
        config, icd_version)
    result["icd_code_complete"] = code
    result["mapping_relation"] = relation
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_robust(path: str, required_columns: list) -> tuple[list, list]:
    """
    Read a CSV tolerating mixed encodings — UTF-8 and cp1252 lines in the same
    file, typical after a partial edit in Excel. Decodes line by line and
    normalizes everything to UTF-8 on write.

    Returns (rows, fieldnames) keeping the original column order and adding
    any missing required column at the end.
    """
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    decoded, fallbacks = [], 0
    for line in raw.split(b"\n"):
        try:
            decoded.append(line.decode("utf-8"))
        except UnicodeDecodeError:
            decoded.append(line.decode("cp1252", errors="replace"))
            fallbacks += 1
    if fallbacks:
        print(f"[CSV] Mixed encoding: {fallbacks} lines decoded as cp1252. "
              f"Normalizing everything to UTF-8 on save.")

    reader = csv.DictReader(decoded)
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
    fieldnames += [c for c in required_columns if c not in fieldnames]
    return rows, fieldnames


def write_csv(path: str, rows: list, fieldnames: list) -> None:
    """Overwrite in place, preserving every original column."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (row.get(k) or "") for k in fieldnames})
    print(f"[CSV] Updated in place: {path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_match_stats(rows: list, labels: list, icd_source: str) -> None:
    """
    Per-run matching summary. `labels` is [(match_type, description), ...],
    which differs between the folders because step 4 is not the same.
    """
    counts = {key: 0 for key, _ in labels}
    sources = {"complete": 0, "core": 0, "none": 0}
    renamed = total = 0

    for row in rows:
        if not (match_type := row.get("match_type")):
            continue
        total += 1
        counts[match_type] = counts.get(match_type, 0) + 1
        if (source := row.get("match_source") or "") in sources:
            sources[source] += 1
        core = (row.get("core_diagnosis") or "").strip()
        if core and normalize(core) != normalize(row.get("diagnosis_en") or ""):
            renamed += 1

    divisor = max(total, 1)
    print("\n── Matching statistics ─────────────────────────────────")
    print(f"  Diagnoses processed         : {total}")
    for key, label in labels:
        value = counts.get(key, 0)
        print(f"  {label:<32}: {value:>3} ({value / divisor * 100:5.1f}%)")

    resolved = total - counts.get("nf", 0)
    print(f"  {'':-<32}   {'':->11}")
    print(f"  {'With ' + icd_source + ' code':<32}: {resolved:>3} "
          f"({resolved / divisor * 100:5.1f}%)")
    print("────────────────────────────────────────────────────────")
    print(f"  Name differs from phrase    : {renamed} of {total}")
    print(f"  Resolved by complete phrase : {sources['complete']}")
    print(f"  Resolved by disease name    : {sources['core']}   <- gain from step 0")
    print("────────────────────────────────────────────────────────\n")
