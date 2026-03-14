"""
Unit tests for NLPService (src/services/nlp_service.py).

Design notes
------------
* NLPService is tested by creating instances via __new__ and injecting mocked
  stop_words / lemmatizer, so NLTK data downloads are skipped entirely.
* `word_tokenize` is patched at the module level for each test that needs it.
* Run with: python3 -m unittest tests.test_nlp_service  (or pytest)
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy NLTK deps so the module can be imported without data downloads.
# sys.modules.setdefault leaves the real module in place if already imported
# (e.g., when running inside Docker with NLTK properly installed).
# ---------------------------------------------------------------------------
_mock_nltk = MagicMock()
_mock_nltk.data.find.return_value = True
_mock_nltk.download = MagicMock()

_mock_corpus = MagicMock()
_mock_corpus.stopwords.words.return_value = [
    "de", "a", "o", "e", "do", "da", "em", "que", "para", "por",
    "com", "um", "uma", "os", "as", "ao",
]

_mock_stem = MagicMock()
_mock_lemmatizer_instance = MagicMock()
_mock_lemmatizer_instance.lemmatize.side_effect = lambda w: w  # identity
_mock_stem.WordNetLemmatizer.return_value = _mock_lemmatizer_instance

_mock_tokenize = MagicMock()
_mock_tokenize.word_tokenize.side_effect = str.split

# Also stub transformers/torch; even though nlp_service.py doesn't use them
# directly, src/services/__init__.py imports classification_service which does.
for _mod, _val in [
    ("nltk", _mock_nltk),
    ("nltk.corpus", _mock_corpus),
    ("nltk.tokenize", _mock_tokenize),
    ("nltk.stem", _mock_stem),
    ("transformers", MagicMock()),
    ("torch", MagicMock()),
]:
    sys.modules.setdefault(_mod, _val)

# Import AFTER stubs are registered
from src.services.nlp_service import NLPService  # noqa: E402


_STOP_WORDS = frozenset([
    "de", "a", "o", "e", "do", "da", "em", "que", "para", "por",
    "com", "um", "uma", "os", "as", "ao",
])


def _make_svc() -> NLPService:
    """Create an NLPService bypassing __init__ to skip NLTK data checks."""
    svc = NLPService.__new__(NLPService)
    svc.language = "portuguese"
    svc.stop_words = _STOP_WORDS
    mock_lem = MagicMock()
    mock_lem.lemmatize.side_effect = lambda w: w
    svc.lemmatizer = mock_lem
    return svc


def _run(svc: NLPService, text: str):
    """Run preprocessar_texto with a deterministic tokenizer stub."""
    with patch("src.services.nlp_service.word_tokenize", side_effect=str.split):
        return svc.preprocessar_texto(text)


# ===========================================================================
class TestPreprocessarTextoEdgeCases(unittest.TestCase):
    """Boundary conditions for preprocessar_texto."""

    def setUp(self):
        self.svc = _make_svc()

    def test_empty_string_returns_empty_text_and_list(self):
        text, tokens = self.svc.preprocessar_texto("")
        self.assertEqual(text, "")
        self.assertEqual(tokens, [])

    def test_none_input_returns_empty_tuple(self):
        text, tokens = self.svc.preprocessar_texto(None)
        self.assertEqual(text, "")
        self.assertEqual(tokens, [])

    def test_whitespace_only_returns_empty_tokens(self):
        _, tokens = _run(self.svc, "   \t  ")
        self.assertEqual(tokens, [])

    def test_return_type_is_tuple_of_str_and_list(self):
        text, tokens = _run(self.svc, "sistema processamento")
        self.assertIsInstance(text, str)
        self.assertIsInstance(tokens, list)


# ===========================================================================
class TestPreprocessarTextoTransformations(unittest.TestCase):
    """Tests for each transformation step applied to input text."""

    def setUp(self):
        self.svc = _make_svc()

    def test_text_converted_to_lowercase(self):
        text, _ = _run(self.svc, "SISTEMA CRÍTICO URGENTE")
        for word in text.split():
            self.assertEqual(word, word.lower())

    def test_https_url_removed(self):
        text, _ = _run(self.svc, "visite https://empresa.com para detalhes")
        self.assertNotIn("https://empresa.com", text)
        self.assertNotIn("https", text)

    def test_http_url_removed(self):
        text, _ = _run(self.svc, "acesse http://portal.interno.com agora")
        self.assertNotIn("http", text)

    def test_www_url_removed(self):
        text, _ = _run(self.svc, "acesse www.empresa.com.br via portal")
        self.assertNotIn("www", text)

    def test_email_address_removed(self):
        text, _ = _run(self.svc, "contato admin@empresa.com urgente")
        self.assertNotIn("@", text)
        self.assertNotIn("admin@empresa.com", text)

    def test_stop_words_not_in_tokens(self):
        _, tokens = _run(self.svc, "o sistema apresentou erro grave de processamento")
        for token in tokens:
            self.assertNotIn(token, _STOP_WORDS,
                             f"Stop word '{token}' should have been removed")

    def test_tokens_with_length_lte_2_are_filtered(self):
        _, tokens = _run(self.svc, "ok de go sistema funcionando")
        for token in tokens:
            self.assertGreater(len(token), 2,
                               f"Short token '{token}' should be filtered out")

    def test_meaningful_words_preserved(self):
        _, tokens = _run(self.svc, "sistema apresentou falha crítica")
        self.assertIn("sistema", tokens)
        self.assertIn("falha", tokens)

    def test_special_characters_removed(self):
        text, _ = _run(self.svc, "sistema! apresentou? falha: crítica.")
        self.assertNotIn("!", text)
        self.assertNotIn("?", text)
        self.assertNotIn(":", text)

    def test_extra_whitespace_collapsed(self):
        text, _ = _run(self.svc, "sistema    apresentou   falha")
        self.assertNotIn("  ", text)

    def test_tokenize_exception_falls_back_to_split(self):
        """word_tokenize failure should fall back to str.split."""
        with patch("src.services.nlp_service.word_tokenize",
                   side_effect=Exception("NLTK unavailable")):
            _, tokens = self.svc.preprocessar_texto("sistema processamento dados")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)


# ===========================================================================
class TestExtrairFeatures(unittest.TestCase):
    """Tests for NLPService.extrair_features."""

    def setUp(self):
        self.svc = _make_svc()

    def _features(self, text: str) -> dict:
        with patch("src.services.nlp_service.word_tokenize", side_effect=str.split):
            return self.svc.extrair_features(text)

    def test_all_required_keys_present(self):
        features = self._features("sistema apresentou falha crítica")
        expected = {
            "original_length", "processed_length", "token_count",
            "unique_tokens", "processed_text", "tokens",
        }
        self.assertEqual(set(features.keys()), expected)

    def test_original_length_equals_raw_input_length(self):
        text = "sistema apresentou falha crítica"
        features = self._features(text)
        self.assertEqual(features["original_length"], len(text))

    def test_token_count_matches_tokens_list_length(self):
        features = self._features("sistema processamento banco dados")
        self.assertEqual(features["token_count"], len(features["tokens"]))

    def test_unique_tokens_at_most_token_count(self):
        features = self._features("sistema sistema falha falha processamento")
        self.assertLessEqual(features["unique_tokens"], features["token_count"])

    def test_tokens_is_list(self):
        features = self._features("sistema processamento")
        self.assertIsInstance(features["tokens"], list)

    def test_empty_text_zero_counts(self):
        with patch("src.services.nlp_service.word_tokenize", side_effect=str.split):
            features = self.svc.extrair_features("")
        self.assertEqual(features["token_count"], 0)
        self.assertEqual(features["unique_tokens"], 0)

    def test_processed_text_is_string(self):
        features = self._features("sistema apresentou falha")
        self.assertIsInstance(features["processed_text"], str)


if __name__ == "__main__":
    unittest.main()
