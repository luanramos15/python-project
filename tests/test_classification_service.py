"""
Unit tests for ClassificationService (src/services/classification_service.py).

Design notes
------------
* `transformers` and `torch` are always stubbed – the real 1.6 GB BART model
  must never be loaded during tests.
* Each test creates a ClassificationService via __new__ with a manually
  configured mock classifier, so the pipeline returns deterministic values.
* Run with: python3 -m unittest tests.test_classification_service  (or pytest)
"""

import sys
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Always mock transformers + torch before any src import.
# ---------------------------------------------------------------------------
for _mod in ("transformers", "torch", "torch.nn", "torch.cuda"):
    sys.modules.setdefault(_mod, MagicMock())

from src.services.classification_service import ClassificationService  # noqa: E402

# ---------------------------------------------------------------------------
# Descriptive labels that ClassificationService uses internally
# ---------------------------------------------------------------------------
_PROD_DESC = (
    "work-related, professional, business communication, project updates, "
    "technical issues, requests for information, official correspondence"
)
_IMPR_DESC = (
    "personal, social, casual conversation, greetings, party invitations, "
    "non-work related topics, entertainment, leisure activities"
)


def _pipeline_result(top: str = "Produtivo") -> dict:
    """Return a fake zero-shot-classification pipeline output dict."""
    if top == "Produtivo":
        return {"labels": [_PROD_DESC, _IMPR_DESC], "scores": [0.85, 0.15]}
    return {"labels": [_IMPR_DESC, _PROD_DESC], "scores": [0.90, 0.10]}


def _make_svc(top: str = "Produtivo") -> ClassificationService:
    """Build a ClassificationService with a mocked pipeline classifier."""
    svc = ClassificationService.__new__(ClassificationService)
    svc.model_name = "facebook/bart-large-mnli"
    svc.category_descriptions = {"Produtivo": _PROD_DESC, "Improdutivo": _IMPR_DESC}
    svc.categories = ["Produtivo", "Improdutivo"]
    svc.classifier = MagicMock(return_value=_pipeline_result(top))
    return svc


# ===========================================================================
class TestClassificarEmailEmptyInput(unittest.TestCase):
    """Edge cases: empty / whitespace-only text."""

    def setUp(self):
        self.svc = _make_svc()

    def test_empty_string_has_error_key(self):
        result = self.svc.classificar_email("")
        self.assertIn("error", result)

    def test_empty_string_default_category_is_improdutivo(self):
        result = self.svc.classificar_email("")
        self.assertEqual(result["category"], "Improdutivo")

    def test_empty_string_confidence_is_zero(self):
        result = self.svc.classificar_email("")
        self.assertEqual(result["confidence"], 0.0)

    def test_whitespace_only_has_error_key(self):
        result = self.svc.classificar_email("   \t\n")
        self.assertIn("error", result)

    def test_empty_text_model_used_is_set(self):
        result = self.svc.classificar_email("")
        self.assertEqual(result["model_used"], "facebook/bart-large-mnli")

    def test_empty_text_classifier_not_called(self):
        self.svc.classificar_email("")
        self.svc.classifier.assert_not_called()


# ===========================================================================
class TestClassificarEmailResultStructure(unittest.TestCase):
    """Shape and type validation for non-empty input."""

    def setUp(self):
        self.svc = _make_svc("Produtivo")

    def test_has_all_required_keys(self):
        result = self.svc.classificar_email("Urgente: erro no servidor de produção")
        for key in ("category", "confidence", "scores", "model_used"):
            self.assertIn(key, result)

    def test_confidence_is_float(self):
        result = self.svc.classificar_email("Relatório de pagamentos com falha")
        self.assertIsInstance(result["confidence"], float)

    def test_confidence_in_0_to_1_range(self):
        result = self.svc.classificar_email("Sistema crítico com falha no processamento")
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_category_is_valid_label(self):
        result = self.svc.classificar_email("Solicitação de suporte técnico")
        self.assertIn(result["category"], ("Produtivo", "Improdutivo"))

    def test_scores_has_both_categories(self):
        result = self.svc.classificar_email("Precisamos de suporte urgente")
        self.assertIn("Produtivo", result["scores"])
        self.assertIn("Improdutivo", result["scores"])

    def test_scores_sum_to_one(self):
        result = self.svc.classificar_email("Suporte técnico urgente necessário")
        total = sum(result["scores"].values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_model_used_matches_config(self):
        result = self.svc.classificar_email("Suporte técnico urgente")
        self.assertEqual(result["model_used"], "facebook/bart-large-mnli")


# ===========================================================================
class TestClassificarEmailCategories(unittest.TestCase):
    """Verify that the pipeline output is mapped to the correct category."""

    def test_produtivo_category_and_confidence(self):
        svc = _make_svc("Produtivo")
        result = svc.classificar_email("Erro crítico no servidor de pagamentos")
        self.assertEqual(result["category"], "Produtivo")
        self.assertAlmostEqual(result["confidence"], 0.85, places=2)

    def test_improdutivo_category_and_confidence(self):
        svc = _make_svc("Improdutivo")
        result = svc.classificar_email("Feliz Natal a todos da equipe!")
        self.assertEqual(result["category"], "Improdutivo")
        self.assertAlmostEqual(result["confidence"], 0.90, places=2)

    def test_produtivo_score_higher_than_improdutivo(self):
        svc = _make_svc("Produtivo")
        result = svc.classificar_email("Sistema crítico com falha")
        self.assertGreater(
            result["scores"]["Produtivo"], result["scores"]["Improdutivo"]
        )

    def test_improdutivo_score_higher_than_produtivo(self):
        svc = _make_svc("Improdutivo")
        result = svc.classificar_email("Boas festas a todos!")
        self.assertGreater(
            result["scores"]["Improdutivo"], result["scores"]["Produtivo"]
        )


# ===========================================================================
class TestClassificarEmailInputHandling(unittest.TestCase):
    """Verify how subject and content are combined and truncated."""

    def test_subject_is_prepended_to_combined_text(self):
        svc = _make_svc("Produtivo")
        svc.classificar_email("Conteúdo do email", subject="Assunto Importante")
        call_args = svc.classifier.call_args[0]
        self.assertIn("Assunto Importante", call_args[0])

    def test_long_text_is_truncated_to_512_chars(self):
        svc = _make_svc("Produtivo")
        long_text = "palavra " * 200  # well over 512 chars
        svc.classificar_email(long_text)
        call_args = svc.classifier.call_args[0]
        self.assertLessEqual(len(call_args[0]), 512)

    def test_empty_subject_still_classifies(self):
        svc = _make_svc("Produtivo")
        result = svc.classificar_email("Conteúdo importante", subject="")
        self.assertNotIn("error", result)


# ===========================================================================
class TestClassificarComDetalhes(unittest.TestCase):
    """Tests for the extended classification method."""

    def setUp(self):
        self.svc = _make_svc("Produtivo")

    def test_has_extra_detail_keys(self):
        result = self.svc.classificar_com_detalhes("Sistema com falha")
        self.assertIn("text_length", result)
        self.assertIn("has_attachment_keywords", result)
        self.assertIn("has_action_keywords", result)

    def test_text_length_is_correct(self):
        text = "Sistema com falha crítica no módulo"
        result = self.svc.classificar_com_detalhes(text)
        self.assertEqual(result["text_length"], len(text))

    def test_action_keyword_urgente_detected(self):
        result = self.svc.classificar_com_detalhes("precisamos de ajuda urgente agora")
        self.assertTrue(result["has_action_keywords"])

    def test_no_action_keyword_in_casual_message(self):
        result = self.svc.classificar_com_detalhes("mensagem de aniversário feliz")
        self.assertFalse(result["has_action_keywords"])

    def test_attachment_keyword_anexo_detected(self):
        result = self.svc.classificar_com_detalhes("segue o documento em anexo para análise")
        self.assertTrue(result["has_attachment_keywords"])

    def test_attachment_keyword_arquivo_detected(self):
        result = self.svc.classificar_com_detalhes("o arquivo foi enviado ontem")
        self.assertTrue(result["has_attachment_keywords"])

    def test_no_attachment_keyword_in_simple_message(self):
        result = self.svc.classificar_com_detalhes("mensagem de boas festas para todos")
        self.assertFalse(result["has_attachment_keywords"])

    def test_base_classification_keys_still_present(self):
        result = self.svc.classificar_com_detalhes("Erro no sistema")
        for key in ("category", "confidence", "scores", "model_used"):
            self.assertIn(key, result)


# ===========================================================================
class TestClassificarEmailPipelineError(unittest.TestCase):
    """Verify that pipeline exceptions are caught and returned gracefully."""

    def test_pipeline_exception_returns_error_dict(self):
        svc = _make_svc()
        svc.classifier.side_effect = RuntimeError("Pipeline unavailable")
        result = svc.classificar_email("Texto qualquer")
        self.assertIn("error", result)
        self.assertIsNone(result["category"])

    def test_pipeline_exception_confidence_zero(self):
        svc = _make_svc()
        svc.classifier.side_effect = RuntimeError("Pipeline unavailable")
        result = svc.classificar_email("Texto qualquer")
        self.assertEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
