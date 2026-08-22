import re


class SemanticChunker:
    """Splits text into sentence-aware chunks with overlap."""

    def __init__(self, target_tokens=400, overlap_tokens=75):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def split_sentences(self, text: str) -> list:
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+')
        sentences = sentence_end.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _pick_chunk_profile(self, text: str) -> tuple[int, int]:
        word_count = len(text.split())
        if word_count > 12000:
            return 900, 40
        if word_count > 6000:
            return 750, 50
        if word_count > 2500:
            return 600, 60
        return self.target_tokens, self.overlap_tokens

    def chunk_text(self, text: str, source_name: str) -> list:
        target_tokens, overlap_tokens = self._pick_chunk_profile(text)
        sentences = self.split_sentences(text)
        chunks = []
        current_sentences = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            approx_tokens = int(len(sentence.split()) * 1.3)
            if current_length + approx_tokens > (target_tokens + 50) and current_sentences:
                chunks.append({
                    "text": " ".join(current_sentences),
                    "metadata": {
                        "source": source_name,
                        "chunk_index": len(chunks),
                        "start_sentence": i - len(current_sentences),
                        "end_sentence": i - 1,
                    },
                })

                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    s_tokens = int(len(s.split()) * 1.3)
                    if overlap_len + s_tokens > overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += s_tokens

                current_sentences = overlap_sentences
                current_length = overlap_len

            current_sentences.append(sentence)
            current_length += approx_tokens

        if current_sentences:
            chunks.append({
                "text": " ".join(current_sentences),
                "metadata": {
                    "source": source_name,
                    "chunk_index": len(chunks),
                    "start_sentence": len(sentences) - len(current_sentences),
                    "end_sentence": len(sentences) - 1,
                },
            })
        return chunks
