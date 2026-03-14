"""
Route-level tests for the Email Classification API (src/routes/email_routes.py).

Design notes
------------
* Uses the Flask application factory with an in-memory SQLite database so the
  full request/response cycle is exercised without external services.
* All AI/ML service calls (classification, NLP, response generation) are patched
  at the route module level so no model is loaded during testing.
* Database rows are wiped between tests to keep each test independent.
* Run with: python3 -m unittest tests.test_routes  (or pytest)
"""

import io
import json
import os
import sys
import unittest
import uuid
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy ML dependencies before any src import
# ---------------------------------------------------------------------------
for _mod in ("transformers", "torch", "torch.nn", "torch.cuda"):
    sys.modules.setdefault(_mod, MagicMock())

os.environ.setdefault("USE_SQLITE", "1")
os.environ.setdefault("SQLITE_DIR", "/tmp")
os.environ.setdefault("HUGGINGFACE_API_TOKEN", "")

from src.app import create_app        # noqa: E402
from src.models.database import db   # noqa: E402

# ---------------------------------------------------------------------------
# Fixed mock payloads returned by stubbed AI services
# ---------------------------------------------------------------------------
_MOCK_CLF = {
    "category": "Produtivo",
    "confidence": 0.92,
    "scores": {"Produtivo": 0.92, "Improdutivo": 0.08},
    "model_used": "facebook/bart-large-mnli",
}
_MOCK_RESP = {
    "response_text": "Prezado(a), agradecemos o contato.",
    "model_used": "template-fallback",
    "subtype": "template",
    "personalization_level": "medium",
}
_MOCK_STATS = {
    "stats": {
        "total_classified": 5,
        "total_corrected": 1,
        "accuracy": 0.8,
        "corrections_by_direction": {"Produtivo → Improdutivo": 1},
    },
    "fine_tuning_readiness": "need_more_data (have 1, need 50)",
    "recommended_approach": "Collect more labeled examples.",
    "human_corrected_pairs": 1,
    "training_pairs_total": 5,
}


def _ai_patches() -> list:
    """Return a list of patch objects that stub all AI/ML service calls."""
    return [
        patch(
            "src.routes.email_routes.classification_service.classificar_email",
            return_value=_MOCK_CLF,
        ),
        patch(
            "src.routes.email_routes.response_service.gerar_resposta",
            return_value=_MOCK_RESP,
        ),
        patch(
            "src.routes.email_routes.nlp_service.preprocessar_texto",
            return_value=("sistema erro pagamento", ["sistema", "erro", "pagamento"]),
        ),
        patch(
            "src.routes.email_routes.training_service.find_correction_for_email",
            return_value=None,
        ),
    ]


# ===========================================================================
# Base test class
# ===========================================================================
class _BaseRouteTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config.update({
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "WTF_CSRF_ENABLED": False,
        })
        with cls.app.app_context():
            db.create_all()

    @classmethod
    def tearDownClass(cls):
        with cls.app.app_context():
            db.drop_all()

    def setUp(self):
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        from src.models.database import Email, Classification, SuggestedResponse
        SuggestedResponse.query.delete()
        Classification.query.delete()
        Email.query.delete()
        db.session.commit()
        self.ctx.pop()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _processar(self, assunto: str = "Assunto Teste",
                   conteudo: str = "Sistema apresentou falha grave no servidor."):
        """POST /api/emails/processar with mocked AI services."""
        patches = _ai_patches()
        for p in patches:
            p.start()
        try:
            rv = self.client.post(
                "/api/emails/processar",
                data=json.dumps({"assunto": assunto, "conteudo": conteudo}),
                content_type="application/json",
            )
        finally:
            for p in patches:
                p.stop()
        return rv

    def _post_json(self, url: str, payload: dict):
        return self.client.post(
            url,
            data=json.dumps(payload),
            content_type="application/json",
        )


# ===========================================================================
class TestHealthEndpoint(_BaseRouteTest):

    def test_returns_200(self):
        rv = self.client.get("/health")
        self.assertEqual(rv.status_code, 200)

    def test_status_healthy(self):
        data = json.loads(self.client.get("/health").data)
        self.assertEqual(data["status"], "healthy")

    def test_database_field_connected(self):
        data = json.loads(self.client.get("/health").data)
        self.assertEqual(data["database"], "connected")

    def test_db_engine_is_sqlite(self):
        data = json.loads(self.client.get("/health").data)
        self.assertEqual(data["db_engine"], "sqlite")


# ===========================================================================
class TestProcessarEndpoint(_BaseRouteTest):

    def test_valid_request_returns_201(self):
        rv = self._processar()
        self.assertEqual(rv.status_code, 201)

    def test_response_has_required_top_level_keys(self):
        data = json.loads(self._processar().data)
        for key in ("email_id", "assunto", "classificacao", "resposta_sugerida", "timestamp"):
            self.assertIn(key, data)

    def test_classificacao_has_required_keys(self):
        clf = json.loads(self._processar().data)["classificacao"]
        for key in ("categoria", "confianca", "scores", "modelo_usado"):
            self.assertIn(key, clf)

    def test_resposta_sugerida_has_required_keys(self):
        resp = json.loads(self._processar().data)["resposta_sugerida"]
        for key in ("texto", "modelo_usado"):
            self.assertIn(key, resp)

    def test_email_id_is_valid_uuid(self):
        email_id = json.loads(self._processar().data)["email_id"]
        uuid.UUID(email_id)  # raises ValueError if not valid

    def test_missing_conteudo_returns_400(self):
        patches = _ai_patches()
        for p in patches:
            p.start()
        try:
            rv = self._post_json("/api/emails/processar", {"assunto": "Sem conteúdo"})
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(rv.status_code, 400)

    def test_empty_conteudo_returns_400(self):
        patches = _ai_patches()
        for p in patches:
            p.start()
        try:
            rv = self._post_json("/api/emails/processar",
                                 {"assunto": "Teste", "conteudo": "   "})
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(rv.status_code, 400)

    def test_no_body_returns_400(self):
        rv = self.client.post("/api/emails/processar", content_type="application/json")
        self.assertEqual(rv.status_code, 400)

    def test_email_persisted_to_database(self):
        from src.models.database import Email
        rv = self._processar(assunto="Persistência DB")
        email_id = json.loads(rv.data)["email_id"]
        email = Email.query.get(email_id)
        self.assertIsNotNone(email)
        self.assertEqual(email.subject, "Persistência DB")

    def test_classification_persisted_to_database(self):
        from src.models.database import Classification
        email_id = json.loads(self._processar().data)["email_id"]
        clf = Classification.query.filter_by(email_id=email_id).first()
        self.assertIsNotNone(clf)
        self.assertEqual(clf.category, "Produtivo")

    def test_suggested_response_persisted_to_database(self):
        from src.models.database import SuggestedResponse
        email_id = json.loads(self._processar().data)["email_id"]
        resp = SuggestedResponse.query.filter_by(email_id=email_id).first()
        self.assertIsNotNone(resp)


# ===========================================================================
class TestListarEmailsEndpoint(_BaseRouteTest):

    def test_empty_database_returns_200(self):
        rv = self.client.get("/api/emails")
        self.assertEqual(rv.status_code, 200)

    def test_response_has_pagination_keys(self):
        data = json.loads(self.client.get("/api/emails").data)
        for key in ("emails", "total", "pages", "current_page", "per_page"):
            self.assertIn(key, data)

    def test_empty_database_total_is_zero(self):
        data = json.loads(self.client.get("/api/emails").data)
        self.assertEqual(data["total"], 0)

    def test_default_page_is_1(self):
        data = json.loads(self.client.get("/api/emails").data)
        self.assertEqual(data["current_page"], 1)

    def test_custom_per_page_respected(self):
        data = json.loads(self.client.get("/api/emails?per_page=5").data)
        self.assertEqual(data["per_page"], 5)

    def test_created_email_appears_in_list(self):
        self._processar(assunto="Listagem Teste")
        data = json.loads(self.client.get("/api/emails").data)
        self.assertEqual(data["total"], 1)

    def test_category_filter_produtivo_matches(self):
        self._processar()  # mock always returns "Produtivo"
        data = json.loads(self.client.get("/api/emails?categoria=Produtivo").data)
        self.assertEqual(data["total"], 1)

    def test_category_filter_improdutivo_returns_zero_for_produtivo_email(self):
        self._processar()
        data = json.loads(self.client.get("/api/emails?categoria=Improdutivo").data)
        self.assertEqual(data["total"], 0)

    def test_multiple_emails_counted(self):
        for _ in range(3):
            self._processar()
        data = json.loads(self.client.get("/api/emails").data)
        self.assertEqual(data["total"], 3)


# ===========================================================================
class TestObterEmailEndpoint(_BaseRouteTest):

    def test_nonexistent_id_returns_404(self):
        rv = self.client.get(f"/api/emails/{uuid.uuid4()}")
        self.assertEqual(rv.status_code, 404)

    def test_valid_id_returns_200(self):
        email_id = json.loads(self._processar().data)["email_id"]
        rv = self.client.get(f"/api/emails/{email_id}")
        self.assertEqual(rv.status_code, 200)

    def test_returned_email_id_matches(self):
        email_id = json.loads(self._processar().data)["email_id"]
        data = json.loads(self.client.get(f"/api/emails/{email_id}").data)
        self.assertEqual(data["id"], email_id)

    def test_returned_data_includes_classificacao(self):
        email_id = json.loads(self._processar().data)["email_id"]
        data = json.loads(self.client.get(f"/api/emails/{email_id}").data)
        self.assertIn("classificacao", data)

    def test_returned_data_includes_respostas(self):
        email_id = json.loads(self._processar().data)["email_id"]
        data = json.loads(self.client.get(f"/api/emails/{email_id}").data)
        self.assertIn("respostas", data)
        self.assertGreater(len(data["respostas"]), 0)


# ===========================================================================
class TestFeedbackEndpoint(_BaseRouteTest):

    def _create_email(self):
        return json.loads(self._processar().data)["email_id"]

    def _feedback(self, email_id: str, payload: dict):
        return self._post_json(f"/api/emails/{email_id}/feedback", payload)

    def _get_response_obj(self, email_id: str):
        from src.models.database import SuggestedResponse
        return SuggestedResponse.query.filter_by(email_id=email_id).first()

    # --- response quality feedback ---

    def test_helpful_feedback_returns_200(self):
        eid = self._create_email()
        resp = self._get_response_obj(eid)
        rv = self._feedback(eid, {"feedback": "helpful", "response_id": resp.id})
        self.assertEqual(rv.status_code, 200)

    def test_not_helpful_feedback_returns_200(self):
        eid = self._create_email()
        resp = self._get_response_obj(eid)
        rv = self._feedback(eid, {"feedback": "not_helpful", "response_id": resp.id})
        self.assertEqual(rv.status_code, 200)

    def test_feedback_persisted_in_database(self):
        eid = self._create_email()
        resp = self._get_response_obj(eid)
        self._feedback(eid, {"feedback": "helpful", "response_id": resp.id})
        db.session.refresh(resp)
        self.assertEqual(resp.user_feedback, "helpful")

    # --- classification correction ---

    def test_classification_correction_returns_200(self):
        eid = self._create_email()
        rv = self._feedback(eid, {"corrected_category": "Improdutivo"})
        self.assertEqual(rv.status_code, 200)

    def test_correction_persisted_in_database(self):
        from src.models.database import Classification
        eid = self._create_email()
        self._feedback(eid, {
            "corrected_category": "Improdutivo",
            "feedback_comment": "AI errou neste caso concreto",
        })
        clf = Classification.query.filter_by(email_id=eid).first()
        self.assertEqual(clf.corrected_category, "Improdutivo")
        self.assertEqual(clf.feedback_comment, "AI errou neste caso concreto")

    def test_combined_feedback_and_correction_returns_200(self):
        eid = self._create_email()
        resp = self._get_response_obj(eid)
        rv = self._feedback(eid, {
            "feedback": "not_helpful",
            "response_id": resp.id,
            "corrected_category": "Improdutivo",
        })
        self.assertEqual(rv.status_code, 200)

    # --- validation ---

    def test_invalid_category_returns_400(self):
        eid = self._create_email()
        rv = self._feedback(eid, {"corrected_category": "Invalido"})
        self.assertEqual(rv.status_code, 400)

    def test_no_actionable_field_returns_400(self):
        eid = self._create_email()
        rv = self._feedback(eid, {"comment": "apenas comentário sem ação"})
        self.assertEqual(rv.status_code, 400)

    def test_nonexistent_email_returns_404(self):
        rv = self._feedback(str(uuid.uuid4()), {"feedback": "helpful"})
        self.assertEqual(rv.status_code, 404)

    def test_response_body_contains_email_id(self):
        eid = self._create_email()
        data = json.loads(self._feedback(eid, {"corrected_category": "Improdutivo"}).data)
        self.assertEqual(data["email_id"], eid)


# ===========================================================================
class TestTrainingStatsEndpoint(_BaseRouteTest):

    def test_returns_200(self):
        with patch(
            "src.routes.email_routes.training_service.fine_tuning_summary",
            return_value=_MOCK_STATS,
        ):
            rv = self.client.get("/api/emails/training/stats")
        self.assertEqual(rv.status_code, 200)

    def test_response_body_has_stats_key(self):
        with patch(
            "src.routes.email_routes.training_service.fine_tuning_summary",
            return_value=_MOCK_STATS,
        ):
            data = json.loads(self.client.get("/api/emails/training/stats").data)
        self.assertIn("stats", data)

    def test_response_body_has_readiness_key(self):
        with patch(
            "src.routes.email_routes.training_service.fine_tuning_summary",
            return_value=_MOCK_STATS,
        ):
            data = json.loads(self.client.get("/api/emails/training/stats").data)
        self.assertIn("fine_tuning_readiness", data)


# ===========================================================================
class TestTrainingExportEndpoint(_BaseRouteTest):

    def test_returns_200(self):
        with patch(
            "src.routes.email_routes.training_service.export_as_jsonl",
            return_value='{"label":"Produtivo"}',
        ):
            rv = self.client.get("/api/emails/training/export")
        self.assertEqual(rv.status_code, 200)

    def test_content_type_is_ndjson(self):
        with patch(
            "src.routes.email_routes.training_service.export_as_jsonl",
            return_value="",
        ):
            rv = self.client.get("/api/emails/training/export")
        self.assertIn("ndjson", rv.content_type)

    def test_only_corrected_param_forwarded(self):
        with patch(
            "src.routes.email_routes.training_service.export_as_jsonl",
            return_value="",
        ) as mock_export:
            self.client.get("/api/emails/training/export?only_corrected=true")
        mock_export.assert_called_once_with(only_corrected=True)

    def test_default_only_corrected_is_false(self):
        with patch(
            "src.routes.email_routes.training_service.export_as_jsonl",
            return_value="",
        ) as mock_export:
            self.client.get("/api/emails/training/export")
        mock_export.assert_called_once_with(only_corrected=False)


# ===========================================================================
class TestUploadEndpoint(_BaseRouteTest):

    def test_no_file_field_returns_400(self):
        rv = self.client.post("/api/emails/upload")
        self.assertEqual(rv.status_code, 400)

    def test_unsupported_extension_returns_400(self):
        rv = self.client.post(
            "/api/emails/upload",
            data={"file": (io.BytesIO(b"conteudo"), "email.doc")},
            content_type="multipart/form-data",
        )
        self.assertEqual(rv.status_code, 400)

    def test_empty_txt_file_returns_400(self):
        patches = _ai_patches()
        for p in patches:
            p.start()
        try:
            rv = self.client.post(
                "/api/emails/upload",
                data={"file": (io.BytesIO(b"   \n  "), "empty.txt")},
                content_type="multipart/form-data",
            )
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(rv.status_code, 400)

    def test_valid_txt_file_returns_201(self):
        content = b"Sistema apresentou falha grave no modulo de pagamentos."
        patches = _ai_patches()
        for p in patches:
            p.start()
        try:
            rv = self.client.post(
                "/api/emails/upload",
                data={
                    "file": (io.BytesIO(content), "email.txt"),
                    "assunto": "Teste Upload",
                },
                content_type="multipart/form-data",
            )
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(rv.status_code, 201)

    def test_txt_upload_response_has_email_id(self):
        content = b"Sistema apresentou falha grave no modulo de pagamentos."
        patches = _ai_patches()
        for p in patches:
            p.start()
        try:
            rv = self.client.post(
                "/api/emails/upload",
                data={"file": (io.BytesIO(content), "email.txt")},
                content_type="multipart/form-data",
            )
        finally:
            for p in patches:
                p.stop()
        data = json.loads(rv.data)
        self.assertIn("email_id", data)


if __name__ == "__main__":
    unittest.main()
