"""
icd10_index.py
--------------
Local ICD-10 catalogue, parsed from the official WHO ClaML XML.

Source file
===========
`icd102019en.xml` — WHO ICD-10 2019 release in ClaML 2.0 format, the same
classification the WHO API serves, but complete and in a single offline
file. Downloaded from the WHO ICD-10 browser (Info > Download area).

Why the XML and not the API
===========================
The WHO ICD-10 API does NOT offer the endpoints the ICD-11 one has:

    GET /icd/release/10/2019/search?q=...      -> 404 (does not exist)
    GET /icd/release/10/2019/codeinfo/A49.1    -> 404 (does not exist)
    GET /icd/release/10/2019/A49.1             -> 200 OK

It can only resolve one code at a time, so building a searchable catalogue
used to require crawling the whole tree (115 s, 32 concurrent workers, WHO
credentials). The ClaML file gives the same tree in 0.7 s, offline, and adds
what the crawl could not provide:

    preferred      11539   official title
    inclusion       9453   SYNONYMS and entry terms
    preferredLong    797   long title carrying the parent context
    exclusion       4798   what does NOT belong in that code
    definition       268

Those synonyms are the point. The crawl only knew C22.0 as "Malignant
neoplasm: Liver cell carcinoma", so a diagnosis reading "Hepatocellular
carcinoma" could not be matched by string similarity and had to be guessed
by the LLM out of a wide, noisy candidate pool. In ClaML, "Hepatocellular
carcinoma" and "Hepatoma" are literal `inclusion` rubrics of C22.0, which
turns that case into an exact match.

Structure exposed
=================
    entities: code -> {
        title, title_long, synonyms, kind, parent,
        chapter_code, chapter_title, depth
    }
    terms:    normalized term -> code      (searchable corpus)

`depth` is the measured distance from the chapter (chapter = 0, block = 1,
category = 2, subcategory = 3, ...), obtained by walking `SuperClass`
upwards. It is not inferred from a table of ranges, which matters because
blocks nest irregularly (C00-D48 contains sub-blocks, so depth is not
constant).

Design decisions worth keeping in mind
======================================
1. Synonyms are indexed ONLY from `kind="category"` classes. In blocks and
   chapters the `inclusion` rubrics are a table of contents, not synonyms:
   block D80-D89 lists "sarcoidosis" among its contents, and indexing that
   would map the diagnosis Sarcoidosis to the block D80-D89 instead of to
   its real category D86.

2. Only categories are assignable. Blocks and chapters are parsed (they are
   needed for the hierarchy) but never enter the searchable corpus, so the
   pipeline can never emit a block as if it were a code.

3. Square brackets are stripped and their content is ALSO indexed as a term
   of its own. ICD-10 puts eponyms and alternative names in brackets —
   "Aortic arch syndrome [Takayasu]", "Giardiasis [lambliasis]" — and 1979
   terms use them. Glued to the word, `[takayasu]` never matches the token
   `takayasu`, so the diagnosis "Takayasu arteritis" retrieved nothing at
   all. Indexed properly it is an exact hit on M31.4.

4. Modifiers are expanded. ClaML keeps the 4th/5th character subdivisions as
   `Modifier` / `ModifierClass` elements instead of pre-expanded codes: N03
   plus modifier S14N00_4 subclass ".5" yields N03.5 "Diffuse
   mesangiocapillary glomerulonephritis". Expanding them recovers the ~1000
   codes the API used to serve pre-expanded, `ExcludeModifier` opt-outs
   included.

Usage
=====
    from icd10_index import ICD10Index

    index = ICD10Index.from_claml("icd102019en.xml")
    index.lookup_exact("hepatocellular carcinoma")   # -> "C22.0"
    index.search("chronic nephritic syndrome ...")   # fuzzy candidate pool
    index.title("C22.0"), index.chapter("C22.0")

As a script, to inspect the parsed catalogue or export it for review:

    python icd10_index.py
    python icd10_index.py --export-json icd10_catalogue.json
"""

import re
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from rapidfuzz import process, fuzz

from icd_common import find_data_file, normalize, strip_dagger

# Default file name, looked up next to the script (see find_data_file).
CLAML_FILENAME = "icd102019en.xml"


# Normalization, dagger stripping and data-file lookup are imported from
# icd_common so both pipelines compare strings identically.

SCRIPT_DIR = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# ClaML PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _label(rubric: ET.Element, drop_references: bool = True) -> str:
    """
    Flatten a <Label> into plain text.

    A Label nests <Para>, <Fragment> and <Reference>, so the text has to be
    walked rather than read from .text. The subtlety is <Reference>: in the
    naming rubrics (preferred, inclusion, preferredLong) it holds the
    dagger/asterisk CROSS-REFERENCE CODE, not part of the name, and a naive
    itertext() glues it to the title:

        <Label>Amoebic liver abscess<Reference usage="aster">K77.0</Reference></Label>
        naive  -> "Amoebic liver abscessK77.0"     wrong
        here   -> "Amoebic liver abscess"          correct

    99 titles were affected. Left glued, the code leaks into the CSV and,
    worse, corrupts the search term so the real name stops matching.

    `drop_references=False` keeps them, which is what instructional rubrics
    need ("Use additional code R57.2 if desired") — there the reference is
    part of the sentence.
    """
    el = rubric.find("Label")
    if el is None:
        return ""

    if not drop_references:
        return " ".join("".join(el.itertext()).split())

    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            if child.tag != "Reference":
                walk(child)
            # A Reference's own text is dropped, but its tail belongs to the
            # surrounding sentence and must survive.
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return " ".join("".join(parts).split())


# Bracket contents shorter than this are notation, not names: ICD-10 writes
# "Potassium [K] excess" and "abnormal gravitation [G] forces", and a
# one-letter term matches every query under partial_ratio.
MIN_BRACKET_CHARS = 3


def _bracket_terms(text: str, max_words: int = 4) -> list:
    """
    Extract eponyms and alternative names written in square brackets.

    ICD-10 uses them heavily: "Aortic arch syndrome [Takayasu]",
    "Giardiasis [lambliasis]". Short contents are real alternative names and
    deserve to be their own search term; long ones are qualifiers, not
    names, so they are skipped — as is chemical/physical notation, which is
    what the length floor filters out.
    """
    out = []
    for alt in re.findall(r"\[([^\]]+)\]", text or ""):
        alt = alt.strip()
        if (alt and len(alt) >= MIN_BRACKET_CHARS
                and len(alt.split()) <= max_words):
            out.append(alt)
    return out


class ICD10Index:
    """Parsed ICD-10 catalogue: hierarchy, titles, synonyms and search."""

    # Terms shorter than this stay searchable by EXACT match but are kept
    # out of the fuzzy corpus. An exact hit on "HIV" or "yaws" is
    # unambiguous; the same term scored with partial_ratio inside a long
    # diagnosis returns 100 for anything that happens to contain it.
    MIN_FUZZY_CHARS = 5

    # partial_ratio rewards a term that is a substring of the query, which is
    # meaningless when the term is far shorter than what is being matched.
    MIN_PARTIAL_LENGTH_RATIO = 0.6

    def __init__(self, entities: dict, terms: dict, source: str = "",
                 version: str = ""):
        self.entities = entities
        self.terms    = terms
        self.source   = source
        self.version  = version
        self._keys    = [t for t in terms if len(t) >= self.MIN_FUZZY_CHARS]

    # ── construction ─────────────────────────────────────────────────────

    @classmethod
    def from_claml(cls, path: str, expand_modifiers: bool = True) -> "ICD10Index":
        root = ET.parse(path).getroot()

        title_el = root.find("Title")
        version  = title_el.get("version", "") if title_el is not None else ""

        entities: dict = {}

        for node in root.findall("Class"):
            code = node.get("code")
            kind = node.get("kind")
            if not code:
                continue

            sup = node.find("SuperClass")
            entry = {
                "title":      "",
                "title_long": None,
                "synonyms":   [],
                "kind":       kind,
                "parent":     sup.get("code") if sup is not None else None,
                "usage":      node.get("usage"),
            }

            for rubric in node.findall("Rubric"):
                rkind = rubric.get("kind")
                value = _label(rubric)
                if not value:
                    continue
                if rkind == "preferred":
                    entry["title"] = value
                elif rkind == "preferredLong":
                    entry["title_long"] = value
                elif rkind == "inclusion" and kind == "category":
                    # Only categories: in blocks/chapters inclusions are a
                    # table of contents, not synonyms.
                    entry["synonyms"].append(value)

            entities[code] = entry

        if expand_modifiers:
            entities.update(_expand_modifiers(root, entities))

        _attach_hierarchy(entities)
        terms = _build_terms(entities)

        return cls(entities, terms, source=str(path), version=version)

    @classmethod
    def load(cls, path: str | None = None,
             expand_modifiers: bool = True) -> "ICD10Index | None":
        """
        Locate and parse the ClaML file. Returns None when it is not found,
        so the caller can degrade instead of crashing.
        """
        resolved = find_data_file(CLAML_FILENAME, path)
        if not resolved:
            return None
        return cls.from_claml(resolved, expand_modifiers=expand_modifiers)

    # ── access ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entities)

    def __contains__(self, code: str) -> bool:
        return strip_dagger(code) in self.entities

    def get(self, code: str) -> dict:
        return self.entities.get(strip_dagger(code), {})

    def title(self, code: str) -> str:
        info = self.get(code)
        return info.get("title_long") or info.get("title", "")

    def chapter(self, code: str) -> str:
        """Chapter as "II - Neoplasms"."""
        info = self.get(code)
        ch_code  = info.get("chapter_code", "")
        ch_title = info.get("chapter_title", "")
        if ch_code and ch_title:
            return f"{ch_code} - {ch_title}"
        return ch_title or ch_code

    def hierarchical_distance(self, code: str) -> int | None:
        """
        Levels between the chapter and this code (chapter = 0, block = 1,
        category = 2, ...). Measured while walking SuperClass, never
        inferred.
        """
        return self.get(code).get("depth")

    def hierarchy_path(self, code: str) -> str:
        """
        The chain of classes that produces `hierarchical_distance`, written
        from chapter down to the code itself:

            "II[chapter] > C00-C97[block] > C00-C75[block] > "
            "C15-C26[block] > C22[category]"

        This makes the distance auditable instead of an unexplained number:
        the count of elements is always distance + 1, and the level tags show
        WHY two categories sit at different depths — C22 hangs off three
        nested blocks while E84 hangs off one, so 4 vs 2 is a property of the
        classification, not an artefact.

        Returns "" when the code is unknown (NF codes included).
        """
        code = strip_dagger(code)
        if code not in self.entities:
            return ""

        chain = []
        cur = code
        seen = set()
        while cur and cur not in seen and len(chain) <= 10:
            info = self.entities.get(cur)
            if not info:
                break
            seen.add(cur)
            chain.append(f"{cur}[{info.get('kind', '?')}]")
            cur = info.get("parent")

        return " > ".join(reversed(chain))

    def is_category(self, code: str) -> bool:
        return self.get(code).get("kind") == "category"

    # ── search ───────────────────────────────────────────────────────────

    def lookup_exact(self, text: str) -> str | None:
        """
        Exact match against the official term or any of its synonyms.
        O(1) dict lookup, fully deterministic — no LLM involved.
        """
        return self.terms.get(normalize(text))

    def search(self, query: str, limit: int = 10, threshold: int = 60,
               per_scorer: int = 8) -> list:
        """
        Fuzzy candidate pool over the official terms.

        Returns a POOL, not an answer: the LLM decides afterwards. Three
        scorers are combined because they fail on different shapes and no
        single one covers the corpus:

          - token_set_ratio : the only one that survives long diagnoses
                              carrying cause and context ("Chronic nephritic
                              syndrome with mesangiocapillary pattern" scores
                              100 against N03)
          - WRatio          : balanced general-purpose default
          - partial_ratio   : rescues the substring case, where the official
                              term wraps around the diagnosis

        Each term keeps its best score across scorers, and the pool is then
        collapsed so a code appears only once — otherwise the same code shows
        up several times through its own synonyms and crowds the pool.

        Returns [{"icd10_code", "title", "score"}, ...], best first.
        """
        query = normalize(query)
        if not query or not self._keys:
            return []

        min_partial_len = len(query) * self.MIN_PARTIAL_LENGTH_RATIO

        by_term: dict[str, float] = {}
        for scorer in (fuzz.token_set_ratio, fuzz.WRatio, fuzz.partial_ratio):
            is_partial = scorer is fuzz.partial_ratio
            for term, score, _ in process.extract(
                query, self._keys, scorer=scorer,
                limit=per_scorer, score_cutoff=threshold,
            ):
                if is_partial and len(term) < min_partial_len:
                    continue          # substring win on a far shorter term
                if score > by_term.get(term, 0):
                    by_term[term] = score

        # Collapse to one entry per code, keeping its best-scoring term
        by_code: dict[str, tuple[float, str]] = {}
        for term, score in by_term.items():
            code = self.terms[term]
            if score > by_code.get(code, (0, ""))[0]:
                by_code[code] = (score, term)

        ranked = sorted(by_code.items(), key=lambda kv: -kv[1][0])[:limit]
        return [
            {"icd10_code": code, "title": self.title(code), "score": round(score, 1)}
            for code, (score, _term) in ranked
        ]

    # ── export (inspection only, never read at runtime) ──────────────────

    def export_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"source": self.source, "version": self.version,
                 "entities": self.entities},
                f, ensure_ascii=False, indent=1,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _expand_modifiers(root: ET.Element, entities: dict) -> dict:
    """
    Expand the 4th/5th character subdivisions ClaML keeps as modifiers.

    A class declares `<ModifiedBy code="S14N00_4"/>`, and the matching
    `<ModifierClass code=".5" modifier="S14N00_4">` carries the label. N03 +
    ".5" therefore yields N03.5 "Diffuse mesangiocapillary
    glomerulonephritis" with its own synonyms. `<ExcludeModifier/>` opts a
    class out of a modifier it would otherwise inherit.

    Only codes that do not already exist are added, so an explicitly spelled
    out subcategory always wins over a generated one.
    """
    # modifier code -> {subclass code -> {title, synonyms}}
    modifier_classes: dict[str, dict] = {}
    for mc in root.findall("ModifierClass"):
        mod  = mc.get("modifier")
        sub  = mc.get("code")
        if not mod or not sub:
            continue
        entry = {"title": "", "synonyms": []}
        for rubric in mc.findall("Rubric"):
            rkind = rubric.get("kind")
            value = _label(rubric)
            if not value:
                continue
            if rkind == "preferred":
                entry["title"] = value
            elif rkind == "inclusion":
                entry["synonyms"].append(value)
        modifier_classes.setdefault(mod, {})[sub] = entry

    generated: dict = {}
    for node in root.findall("Class"):
        code = node.get("code")
        if not code or node.get("kind") != "category":
            continue

        excluded = {e.get("code") for e in node.findall("ExcludeModifier")}
        for mb in node.findall("ModifiedBy"):
            mod = mb.get("code")
            if not mod or mod in excluded or mod not in modifier_classes:
                continue

            for sub, entry in modifier_classes[mod].items():
                # Subclass codes appear as ".5" or as "5"
                suffix = sub if sub.startswith(".") else f".{sub}"
                new_code = f"{code}{suffix}" if "." not in code else f"{code}{sub.lstrip('.')}"
                if new_code in entities or new_code in generated:
                    continue
                generated[new_code] = {
                    "title":      entry["title"],
                    "title_long": None,
                    "synonyms":   list(entry["synonyms"]),
                    "kind":       "category",
                    "parent":     code,
                    "usage":      None,
                }

    return generated


def _attach_hierarchy(entities: dict) -> None:
    """
    Walk SuperClass upwards for every entity to attach the chapter it
    belongs to and its depth below that chapter.
    """
    for code, info in entities.items():
        depth, cur = 0, code
        seen = {cur}
        while True:
            parent = entities.get(cur, {}).get("parent")
            if not parent or parent in seen or depth >= 10:
                break
            cur = parent
            seen.add(cur)
            depth += 1
        info["depth"]         = depth
        info["chapter_code"]  = cur
        info["chapter_title"] = entities.get(cur, {}).get("title", "")


def _build_terms(entities: dict) -> dict:
    """
    Build the searchable corpus: normalized term -> code.

    Only categories are indexed, because only categories are assignable.
    Each entity contributes its official title, its synonyms, and any short
    eponym written in brackets inside either of those.

    The first term to claim a normalized string keeps it, so a title never
    loses to a synonym of another code.
    """
    terms: dict[str, str] = {}
    for code, info in entities.items():
        if info.get("kind") != "category":
            continue
        for source_text in [info.get("title", ""), *info.get("synonyms", [])]:
            if not source_text:
                continue
            key = normalize(source_text)
            if key:
                terms.setdefault(key, code)
            for alt in _bracket_terms(source_text):
                alt_key = normalize(alt)
                if alt_key:
                    terms.setdefault(alt_key, code)
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspect the ICD-10 catalogue parsed from the WHO ClaML XML"
    )
    parser.add_argument("--claml", default=None,
                        help=f"Path to {CLAML_FILENAME}. If omitted it is "
                             f"searched next to the script and in its parent "
                             f"folders.")
    parser.add_argument("--export-json", default=None,
                        help="Write the parsed catalogue to JSON for manual "
                             "inspection. Nothing reads it at runtime.")
    parser.add_argument("--no-modifiers", action="store_true",
                        help="Skip expanding the 4th/5th character modifiers")
    args = parser.parse_args()

    path = find_data_file(CLAML_FILENAME, args.claml)
    if not path:
        print(f"[ERROR] {CLAML_FILENAME} not found starting from {SCRIPT_DIR}")
        return

    index = ICD10Index.from_claml(path, expand_modifiers=not args.no_modifiers)
    kinds: dict[str, int] = {}
    for info in index.entities.values():
        kinds[info.get("kind", "?")] = kinds.get(info.get("kind", "?"), 0) + 1

    print(f"[ICD-10] Source : {index.source}")
    print(f"[ICD-10] Version: {index.version}")
    print(f"[ICD-10] {len(index)} classes {kinds}")
    print(f"[ICD-10] {len(index.terms)} searchable terms\n")

    for probe in ["Hepatocellular carcinoma", "Sarcoidosis", "Takayasu arteritis"]:
        code = index.lookup_exact(probe)
        print(f"  exact  {probe:<26} -> {code}  {index.title(code) if code else ''}")

    print()
    for probe in ["C22.0", "D86", "N03.5"]:
        print(f"  {probe:<8} {index.title(probe)[:46]:<46} | "
              f"{index.chapter(probe)[:34]:<34} | depth {index.hierarchical_distance(probe)}")

    if args.export_json:
        index.export_json(args.export_json)
        print(f"\n[ICD-10] Exported to {args.export_json}")


if __name__ == "__main__":
    main()
