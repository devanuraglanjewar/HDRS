import re

# Common abbreviations to avoid false sentence boundaries
COMMON_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "fig", "al", "e.g", "i.e",
    "no", "vol", "pp", "dept", "inc", "ltd", "corp", "co", "approx", "approx.", "est",
}


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting common abbreviations, bullet points,
    and code/formula boundaries.
    """
    if not text or not text.strip():
        return []

    cleaned = text.strip()
    # Normalize excessive horizontal spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # First, split by explicit line breaks if they separate distinct statements/bullets
    raw_lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

    sentences = []
    for line in raw_lines:
        # Check if line is a bullet or numbered list item
        bullet_match = re.match(r"^(\d+[\.\)]|[\*\-\•])\s*(.+)$", line)
        if bullet_match:
            content = bullet_match.group(2).strip()
            # Split inside the bullet if it contains multiple sentences
            sub_sents = _split_line_into_sentences(content)
            sentences.extend(sub_sents)
        else:
            sub_sents = _split_line_into_sentences(line)
            sentences.extend(sub_sents)

    # Filter out empty or whitespace-only items
    final_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
    return final_sentences if final_sentences else ([cleaned] if cleaned else [])


def _split_line_into_sentences(line: str) -> list[str]:
    if not line:
        return []

    # Regex splits on period, exclamation, question mark followed by space or end
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])", line)
    
    merged = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer} {part}".strip() if buffer else part
        # Check if last word before punctuation was an abbreviation
        last_word_match = re.search(r"\b([A-Za-z]+)\.$", candidate)
        if last_word_match and last_word_match.group(1).lower() in COMMON_ABBREVIATIONS:
            buffer = candidate
        else:
            merged.append(candidate)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)

    return merged
