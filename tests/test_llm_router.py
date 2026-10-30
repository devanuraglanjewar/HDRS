import unittest
from unittest.mock import patch

from src.core import llm


class TestLLMRouter(unittest.TestCase):
    def setUp(self):
        for state in llm._provider_state.values():
            state["degraded_until"] = 0.0
            state["last_error"] = ""

    def test_auto_uses_gemini_first(self):
        with patch.object(llm, "_provider_is_configured", return_value=True), \
             patch.object(llm, "generate_with_gemini", return_value="gemini answer") as gemini, \
             patch.object(llm, "generate_with_nemotron") as nemotron, \
             patch.object(llm, "generate_with_ollama") as ollama:
            result = llm.call_with_fallback("p", "s", "auto")

        self.assertEqual(result, ("gemini answer", "gemini", "gemini-3.6-flash"))
        gemini.assert_called_once()
        nemotron.assert_not_called()
        ollama.assert_not_called()

    def test_auto_falls_back_to_nemotron_then_ollama(self):
        calls = []

        def gemini(*args):
            calls.append("gemini")
            raise RuntimeError("429 quota exceeded")

        def nemotron(*args):
            calls.append("nemotron")
            raise RuntimeError("503 unavailable")

        def ollama(*args):
            calls.append("ollama")
            return "local answer"

        with patch.object(llm, "_provider_is_configured", return_value=True), \
             patch.object(llm, "generate_with_gemini", side_effect=gemini), \
             patch.object(llm, "generate_with_nemotron", side_effect=nemotron), \
             patch.object(llm, "generate_with_ollama", side_effect=ollama):
            result = llm.call_with_fallback("p", "s", "auto")

        self.assertEqual(result[0], "local answer")
        self.assertEqual(result[1], "ollama")
        self.assertEqual(calls, ["gemini", "nemotron", "ollama"])

    def test_quota_failure_temporarily_skips_provider(self):
        with patch.object(llm, "_provider_is_configured", return_value=True), \
             patch.object(llm, "generate_with_gemini", side_effect=RuntimeError("429 quota exceeded")), \
             patch.object(llm, "generate_with_nemotron", return_value="nemotron answer") as nemotron:
            first = llm.call_with_fallback("p", "s", "auto")
            second = llm.call_with_fallback("p", "s", "auto")

        self.assertEqual(first[1], "nemotron")
        self.assertEqual(second[1], "nemotron")
        self.assertEqual(nemotron.call_count, 2)


if __name__ == "__main__":
    unittest.main()
