import re
from collections import defaultdict
from typing import List, Dict, Tuple

from langchain.schema import Document

JARGON_MAP = {
    r"\bdisrupt(?:ing|ed|s|ion|ive|or)?\b": "innovate",
    r"\bdominate(?:s|d|ing|ion)?\b": "lead the market",
    r"\bcrush(?:ing|ed|es)?\b": "outperform",
    r"\bkill(?:er|ing|ed)?\b": "strategic",
    r"\bdestroy(?:ing|ed|s)?\b": "transform",
    r"\battack(?:ing|ed|s)?\b": "enter",
    r"\baggressive(?:ly)?\b": "strategic",
    r"\bwar\b": "competitive landscape",
    r"\bmonopolize?\b": "scale",
    r"\bcapture the market\b": "grow market share",
    r"\bwipe out\b": "outcompete",
    r"\bunleash\b": "deploy",
    r"\bbattle\b": "compete",
    r"\barms race\b": "capability investment",
    r"\bdominance\b": "leadership",
    r"\bconquer\b": "win",
    r"\bobliterate\b": "outpace",
    r"\bdevastating\b": "impactful",
}

STOP_WORDS = {
    "the", "a", "an", "in", "at", "of", "to", "for", "is", "are",
    "was", "were", "be", "been", "by", "with", "from", "as", "on",
    "this", "that", "these", "those", "have", "has", "had", "will",
    "would", "could", "should", "may", "can", "which", "their", "also",
    "more", "than", "about", "into", "over", "each", "such", "when",
}


def rewrite_jargon(text: str) -> Tuple[str, List[Dict]]:
    """Rewrite aggressive market jargon into preferred strategic terminology."""
    changes = []
    result = text

    for pattern, replacement in JARGON_MAP.items():
        matches = list(re.finditer(pattern, result, re.IGNORECASE))
        if matches:
            for match in matches:
                start = max(0, match.start() - 35)
                end = min(len(result), match.end() + 35)
                changes.append({
                    "original": match.group(),
                    "replacement": replacement,
                    "context": result[start:end].strip(),
                })
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result, changes


def _get_keywords(text: str):
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def _extract_numbers(text: str) -> List[float]:
    matches = re.findall(
        r"\b(\d+(?:\.\d+)?)\s*(?:billion|million|thousand|percent|%|k|b|m)?\b",
        text,
        re.IGNORECASE,
    )
    results = []
    for m in matches:
        try:
            v = float(m)
            if v > 0:
                results.append(v)
        except ValueError:
            pass
    return results


def detect_contradictions(docs: List[Document]) -> List[Dict]:
    """Find potentially contradictory numerical claims across source documents."""
    by_source = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", str(id(doc)))
        by_source[source].append(doc)

    sources = list(by_source.keys())
    if len(sources) < 2:
        return []

    contradictions = []
    seen_pairs: set = set()

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            src_a, src_b = sources[i], sources[j]

            for chunk_a in by_source[src_a]:
                kw_a = _get_keywords(chunk_a.page_content)
                nums_a = _extract_numbers(chunk_a.page_content)
                if len(kw_a) < 5 or not nums_a:
                    continue

                for chunk_b in by_source[src_b]:
                    kw_b = _get_keywords(chunk_b.page_content)
                    nums_b = _extract_numbers(chunk_b.page_content)
                    if len(kw_b) < 5 or not nums_b:
                        continue

                    overlap = len(kw_a & kw_b) / max(min(len(kw_a), len(kw_b)), 1)
                    if overlap < 0.25:
                        continue

                    for na in nums_a:
                        for nb in nums_b:
                            if na <= 0 or nb <= 0:
                                continue
                            ratio = min(na, nb) / max(na, nb)
                            if 0.15 <= ratio <= 0.75:
                                pair_key = (src_a, src_b, round(na, 1), round(nb, 1))
                                if pair_key in seen_pairs:
                                    continue
                                seen_pairs.add(pair_key)
                                diff_pct = abs(na - nb) / max(na, nb)
                                contradictions.append({
                                    "source_a": src_a,
                                    "source_b": src_b,
                                    "text_a": chunk_a.page_content[:300],
                                    "text_b": chunk_b.page_content[:300],
                                    "value_a": na,
                                    "value_b": nb,
                                    "keyword_overlap": round(overlap, 2),
                                    "difference_pct": round(diff_pct * 100, 1),
                                    "severity": "high" if diff_pct > 0.4 else "medium",
                                })

    contradictions.sort(
        key=lambda x: (x["severity"] == "high", x["difference_pct"]),
        reverse=True,
    )
    return contradictions[:10]
