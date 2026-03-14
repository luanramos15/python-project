"""
Unit tests for ResponseService (src/services/response_service.py).

Design notes
------------
* `requests.post` is always mocked so no real HTTP is made.
* ResponseService instances are created fresh per test class to isolate
  env variable settings (API token present vs. absent).
* Run with: python3 -m unittest tests.test_response_service  (or pytest)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock transformers/torch so src/services/__init__.py doesn't try to load the
# BART model during the transitive import of classification_service.
# ---------------------------------------------------------------------------
for _mod in ("transformers", "torch", "torch.nn"):
    sys.modules.setdefault(_mod, MagicMock())

from src.services.response_service import ResponseService  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_svc(token: str = "", model: str = "Qwen/Qwen2.5-72B-Instruct") -> ResponseService:
    """Create a fresh ResponseService with the given env configuration."""
    env = {"HUGGINGFACE_API_TOKEN": token, "HUGGINGFACE_MODEL_RESPONSE": model}
    with patch.dict(os.environ, env, clear=False):
        return ResponseService()


def _ok_api_response(text: str) -> MagicMock:
    """Return a mocked requests.Response for a successful API call."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


_LONG_TEXT = (
    "Prezado(a), agradecemos o contato referente ao seu problema. "
    "Nossa equipe está analisando e retornará em breve."
)  # > 20 chars → treated as valid


# ===========================================================================
class TestResponseServiceNoToken(unittest.TestCase):
    """Behaviour when HUGGINGFACE_API_TOKEN is empty/absent."""

    def setUp(self):
        self.svc = _make_svc(token="")

    def test_gerar_resposta_returns_template_subtype(self):
        result = self.svc.gerar_resposta("Erro no sistema", "Urgente")
        self.assertEqual(result["subtype"], "template")

    def test_gerar_resposta_model_is_fallback(self):
        result = self.svc.gerar_resposta("Erro no sistema", "Urgente")
        self.assertEqual(result["model_used"], "template-fallback")

    def test_api_not_called_without_token(self):
        with patch("requests.post") as mock_post:
            self.svc.gerar_resposta("Conteúdo", "Assunto")
        mock_post.assert_not_called()

    def test_template_response_contains_subject(self):
        result = self.svc.gerar_resposta("Conteúdo qualquer", "Pedido de Suporte")
        self.assertIn("Pedido de Suporte", result["response_text"])

    def test_template_produtivo_and_improdutivo_differ(self):
        prod = self.svc._template_response("Assunto", "Produtivo")["response_text"]
        impr = self.svc._template_response("Assunto", "Improdutivo")["response_text"]
        self.assertNotEqual(prod, impr)

    def test_template_empty_subject_uses_default_phrase(self):
        result = self.svc._template_response(subject="", category="Produtivo")
        self.assertIn("sua mensagem", result["response_text"])

    def test_template_response_has_all_required_keys(self):
        result = self.svc._template_response("Suporte", "Produtivo")
        for key in ("response_text", "model_used", "subtype", "personalization_level"):
            self.assertIn(key, result)


# ===========================================================================
class TestResponseServiceAPISuccess(unittest.TestCase):
    """Behaviour when the HF Inference API call succeeds."""

    def setUp(self):
        self.svc = _make_svc(token="hf_testtoken123")

    def test_returns_ai_generated_subtype(self):
        with patch("requests.post", return_value=_ok_api_response(_LONG_TEXT)):
            result = self.svc.gerar_resposta("Erro no módulo", "Urgente", "Produtivo")
        self.assertEqual(result["subtype"], "ai_generated")

    def test_response_text_matches_api_output(self):
        with patch("requests.post", return_value=_ok_api_response(_LONG_TEXT)):
            result = self.svc.gerar_resposta("Erro no módulo", "Urgente", "Produtivo")
        self.assertEqual(result["response_text"], _LONG_TEXT)

    def test_model_used_is_set_from_config(self):
        with patch("requests.post", return_value=_ok_api_response(_LONG_TEXT)):
            result = self.svc.gerar_resposta("Erro no módulo", "Urgente", "Produtivo")
        self.assertEqual(result["model_used"], "Qwen/Qwen2.5-72B-Instruct")

    def test_personalization_level_is_high(self):
        with patch("requests.post", return_value=_ok_api_response(_LONG_TEXT)):
            result = self.svc.gerar_resposta("Erro no módulo", "Urgente", "Produtivo")
        self.assertEqual(result["personalization_level"], "high")

    def test_api_called_with_bearer_token(self):
        with patch("requests.post", return_value=_ok_api_response(_LONG_TEXT)) as mp:
            self.svc.gerar_resposta("Conteúdo", "Assunto", "Produtivo")
        headers = mp.call_args[1]["headers"]
        self.assertIn("Authorization", headers)
        self.assertTrue(headers["Authorization"].startswith("Bearer "))


# ===========================================================================
class TestResponseServiceAPIFallback(unittest.TestCase):
    """API failure scenarios should fall back to the template."""

    def setUp(self):
        self.svc = _make_svc(token="hf_testtoken123")

    def test_http_503_triggers_template_fallback(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("requests.post", return_value=mock_resp):
            result = self.svc.gerar_resposta("Conteúdo", "Assunto", "Produtivo")
        self.assertEqual(result["subtype"], "template")

    def test_connection_error_triggers_template_fallback(self):
        with patch("requests.post", side_effect=ConnectionError("timeout")):
            result = self.svc.gerar_resposta("Conteúdo", "Assunto", "Produtivo")
        self.assertEqual(result["subtype"], "template")

    def test_short_api_response_triggers_template_fallback(self):
        """Generated text shorter than 20 chars is treated as invalid."""
        with patch("requests.post", return_value=_ok_api_response("OK")):
            result = self.svc.gerar_resposta("Conteúdo", "Assunto", "Produtivo")
        self.assertEqual(result["subtype"], "template")

    def test_empty_choices_triggers_template_fallback(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_resp):
            result = self.svc.gerar_resposta("Conteúdo", "Assunto", "Produtivo")
        self.assertEqual(result["subtype"], "template")

    def test_http_exception_triggers_template_fallback(self):
        import requests as req_mod
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_mod.HTTPError("500 Server Error")
        mock_resp.status_code = 500
        with patch("requests.post", return_value=mock_resp):
            result = self.svc.gerar_resposta("Conteúdo", "Assunto", "Produtivo")
        self.assertEqual(result["subtype"], "template")


# ===========================================================================
class TestBuildMessages(unittest.TestCase):
    """Tests for the internal message-building helper."""

    def setUp(self):
        self.svc = _make_svc(token="hf_testtoken")

    def test_returns_two_strings(self):
        result = self.svc._build_messages("Conteúdo", "Assunto", "Produtivo")
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], str)
        self.assertIsInstance(result[1], str)

    def test_produtivo_user_message_is_professional(self):
        _, user_msg = self.svc._build_messages("Conteúdo", "Assunto", "Produtivo")
        self.assertIn("profissional", user_msg.lower())

    def test_improdutivo_user_message_is_informal(self):
        _, user_msg = self.svc._build_messages("Conteúdo", "Assunto", "Improdutivo")
        self.assertIn("informal", user_msg.lower())

    def test_subject_included_in_user_message(self):
        _, user_msg = self.svc._build_messages("Conteúdo", "Meu Assunto", "Produtivo")
        self.assertIn("Meu Assunto", user_msg)

    def test_empty_subject_shows_placeholder(self):
        _, user_msg = self.svc._build_messages("Conteúdo", "", "Produtivo")
        self.assertIn("sem assunto", user_msg.lower())

    def test_long_content_truncated_in_user_message(self):
        long_content = "x" * 3000
        _, user_msg = self.svc._build_messages(long_content, "Assunto", "Produtivo")
        # Content is capped at 1500 chars; total message must be reasonable
        self.assertLessEqual(len(user_msg), 2000)

    def test_system_message_mentions_portuguese(self):
        sys_msg, _ = self.svc._build_messages("Conteúdo", "Assunto", "Produtivo")
        self.assertIn("português", sys_msg.lower())


if __name__ == "__main__":
    unittest.main()
