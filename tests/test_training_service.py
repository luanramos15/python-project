"""
Unit/integration tests for TrainingService (src/services/training_service.py).

Design notes
------------
* Uses a real Flask app configured with an in-memory SQLite database so that
  SQLAlchemy models work exactly as in production.
* Each test method starts with a clean database (rows are wiped in tearDown).
* `transformers` and `torch` are mocked before any src import so that the
  ClassificationService singleton is created without loading the real model.
* Run with: python3 -m unittest tests.test_training_service  (or pytest)
"""

import json
import os
import sys
import unittest
import uuid
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub heavy ML deps before any src import
# ---------------------------------------------------------------------------
for _mod in ("transformers", "torch", "torch.nn", "torch.cuda"):
    sys.modules.setdefault(_mod, MagicMock())

os.environ.setdefault("USE_SQLITE", "1")
os.environ.setdefault("SQLITE_DIR", "/tmp")
os.environ.setdefault("HUGGINGFACE_API_TOKEN", "")

from src.app import create_app              # noqa: E402
from src.models.database import db          # noqa: E402
from src.services.training_service import TrainingService  # noqa: E402


# ===========================================================================
# Shared test-app fixture (created once per module)
# ===========================================================================
def _make_app():
    app = create_app()
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    })
    return app


class _DBTestCase(unittest.TestCase):
    """Base class that provides a fresh Flask + SQLite context per test."""

    @classmethod
    def setUpClass(cls):
        cls.app = _make_app()
        with cls.app.app_context():
            db.create_all()

    @classmethod
    def tearDownClass(cls):
        with cls.app.app_context():
            db.drop_all()

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        self.svc = TrainingService()

    def tearDown(self):
        from src.models.database import Email, Classification, SuggestedResponse
        SuggestedResponse.query.delete()
        Classification.query.delete()
        Email.query.delete()
        db.session.commit()
        self.ctx.pop()

    # ------------------------------------------------------------------
    # Helper: insert one email + classification row
    # ------------------------------------------------------------------
    def _insert(
        self,
        category: str,
        *,
        corrected_category: str | None = None,
        feedback_comment: str | None = None,
        confidence: float = 0.9,
        subject: str | None = None,
        content: str | None = None,
    ):
        from src.models.database import Email, Classification

        eid = str(uuid.uuid4())
        email = Email(
            id=eid,
            subject=subject or f"Subject {eid[:8]}",
            content=content or f"Content {eid[:8]}",
        )
        db.session.add(email)
        db.session.flush()

        clf = Classification(
            id=str(uuid.uuid4()),
            email_id=eid,
            category=category,
            confidence=confidence,
            model_used="facebook/bart-large-mnli",
            corrected_category=corrected_category,
            feedback_comment=feedback_comment,
        )
        db.session.add(clf)
        db.session.commit()
        return email, clf


# ===========================================================================
class TestGetFeedbackStats(_DBTestCase):

    def test_empty_db_accuracy_is_none(self):
        stats = self.svc.get_feedback_stats()
        self.assertIsNone(stats["accuracy"])
        self.assertEqual(stats["total_classified"], 0)
        self.assertEqual(stats["total_corrected"], 0)

    def test_no_corrections_accuracy_is_one(self):
        for _ in range(4):
            self._insert("Produtivo")
        stats = self.svc.get_feedback_stats()
        self.assertEqual(stats["total_classified"], 4)
        self.assertEqual(stats["total_corrected"], 0)
        self.assertAlmostEqual(stats["accuracy"], 1.0)

    def test_one_correction_reduces_accuracy(self):
        self._insert("Produtivo", corrected_category="Improdutivo")
        self._insert("Produtivo")
        self._insert("Produtivo")
        self._insert("Produtivo")
        stats = self.svc.get_feedback_stats()
        self.assertEqual(stats["total_classified"], 4)
        self.assertEqual(stats["total_corrected"], 1)
        self.assertAlmostEqual(stats["accuracy"], 0.75, places=4)

    def test_all_corrected_accuracy_is_zero(self):
        for _ in range(3):
            self._insert("Produtivo", corrected_category="Improdutivo")
        stats = self.svc.get_feedback_stats()
        self.assertAlmostEqual(stats["accuracy"], 0.0)

    def test_correction_direction_counted(self):
        self._insert("Produtivo", corrected_category="Improdutivo")
        self._insert("Produtivo", corrected_category="Improdutivo")
        self._insert("Improdutivo", corrected_category="Produtivo")
        stats = self.svc.get_feedback_stats()
        directions = stats["corrections_by_direction"]
        self.assertEqual(directions["Produtivo → Improdutivo"], 2)
        self.assertEqual(directions["Improdutivo → Produtivo"], 1)

    def test_stats_has_all_expected_keys(self):
        stats = self.svc.get_feedback_stats()
        for key in ("total_classified", "total_corrected", "accuracy",
                    "corrections_by_direction"):
            self.assertIn(key, stats)


# ===========================================================================
class TestGetTrainingPairs(_DBTestCase):

    def test_empty_db_returns_empty_list(self):
        pairs = self.svc.get_training_pairs()
        self.assertEqual(pairs, [])

    def test_returns_one_pair_per_email(self):
        self._insert("Produtivo")
        self._insert("Improdutivo")
        pairs = self.svc.get_training_pairs()
        self.assertEqual(len(pairs), 2)

    def test_pair_has_required_keys(self):
        self._insert("Produtivo")
        pairs = self.svc.get_training_pairs()
        self.assertEqual(len(pairs), 1)
        for key in ("email_id", "subject", "content", "label",
                    "original_label", "was_corrected", "source"):
            self.assertIn(key, pairs[0])

    def test_corrected_pair_label_is_corrected_category(self):
        self._insert("Produtivo", corrected_category="Improdutivo")
        pairs = self.svc.get_training_pairs()
        self.assertEqual(pairs[0]["label"], "Improdutivo")
        self.assertTrue(pairs[0]["was_corrected"])
        self.assertEqual(pairs[0]["source"], "human_correction")

    def test_confirmed_pair_label_is_original_category(self):
        self._insert("Produtivo")
        pairs = self.svc.get_training_pairs()
        self.assertEqual(pairs[0]["label"], "Produtivo")
        self.assertFalse(pairs[0]["was_corrected"])
        self.assertEqual(pairs[0]["source"], "model_confirmed")

    def test_only_corrected_filter(self):
        self._insert("Produtivo")                                          # not corrected
        self._insert("Produtivo", corrected_category="Improdutivo")        # corrected
        pairs = self.svc.get_training_pairs(only_corrected=True)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["was_corrected"])

    def test_confidence_field_present(self):
        self._insert("Produtivo", confidence=0.77)
        pairs = self.svc.get_training_pairs()
        self.assertAlmostEqual(pairs[0]["confidence"], 0.77, places=2)


# ===========================================================================
class TestExportAsJsonl(_DBTestCase):

    def test_empty_db_returns_empty_string(self):
        result = self.svc.export_as_jsonl()
        self.assertEqual(result.strip(), "")

    def test_returns_string(self):
        self._insert("Produtivo")
        result = self.svc.export_as_jsonl()
        self.assertIsInstance(result, str)

    def test_each_line_is_valid_json(self):
        self._insert("Produtivo")
        self._insert("Improdutivo", corrected_category="Produtivo")
        jsonl = self.svc.export_as_jsonl()
        for line in jsonl.strip().split("\n"):
            parsed = json.loads(line)
            self.assertIsInstance(parsed, dict)

    def test_only_corrected_flag_filters_lines(self):
        self._insert("Produtivo")                                       # not corrected
        self._insert("Produtivo", corrected_category="Improdutivo")     # corrected
        jsonl = self.svc.export_as_jsonl(only_corrected=True)
        lines = [l for l in jsonl.strip().split("\n") if l]
        self.assertEqual(len(lines), 1)


# ===========================================================================
class TestFindCorrectionForEmail(_DBTestCase):

    def test_exact_match_returns_override(self):
        self._insert(
            "Produtivo",
            corrected_category="Improdutivo",
            subject="Assunto Único",
            content="Conteúdo Específico do Email",
        )
        result = self.svc.find_correction_for_email(
            "Assunto Único", "Conteúdo Específico do Email"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["category"], "Improdutivo")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(result["source"], "human_correction")

    def test_no_match_returns_none(self):
        self._insert("Produtivo", corrected_category="Improdutivo")
        result = self.svc.find_correction_for_email("Outro Assunto", "Outro Conteúdo")
        self.assertIsNone(result)

    def test_case_insensitive_match(self):
        self._insert(
            "Produtivo",
            corrected_category="Improdutivo",
            subject="ASSUNTO MAIÚSCULO",
            content="CONTEÚDO EM MAIÚSCULO",
        )
        result = self.svc.find_correction_for_email(
            "assunto maiúsculo", "conteúdo em maiúsculo"
        )
        self.assertIsNotNone(result)

    def test_no_corrected_category_returns_none(self):
        """An email without a human correction must not be returned."""
        self._insert(
            "Produtivo",
            corrected_category=None,
            subject="Assunto Qualquer",
            content="Conteúdo Qualquer",
        )
        result = self.svc.find_correction_for_email(
            "Assunto Qualquer", "Conteúdo Qualquer"
        )
        self.assertIsNone(result)

    def test_whitespace_normalized_in_match(self):
        self._insert(
            "Produtivo",
            corrected_category="Improdutivo",
            subject="  Assunto Com Espaços  ",
            content="  Conteúdo Com Espaços  ",
        )
        result = self.svc.find_correction_for_email(
            "Assunto Com Espaços", "Conteúdo Com Espaços"
        )
        self.assertIsNotNone(result)


# ===========================================================================
class TestFineTuningSummary(_DBTestCase):

    def test_has_all_expected_keys(self):
        summary = self.svc.fine_tuning_summary()
        for key in ("stats", "fine_tuning_readiness", "recommended_approach",
                    "human_corrected_pairs", "training_pairs_total"):
            self.assertIn(key, summary)

    def test_not_ready_when_below_threshold(self):
        for _ in range(10):
            self._insert("Produtivo", corrected_category="Improdutivo",
                         subject=str(uuid.uuid4()), content=str(uuid.uuid4()))
        summary = self.svc.fine_tuning_summary()
        self.assertNotEqual(summary["fine_tuning_readiness"], "ready")
        self.assertIn("need_more_data", summary["fine_tuning_readiness"])

    def test_ready_when_at_or_above_threshold(self):
        for _ in range(50):
            self._insert("Produtivo", corrected_category="Improdutivo",
                         subject=str(uuid.uuid4()), content=str(uuid.uuid4()))
        summary = self.svc.fine_tuning_summary()
        self.assertEqual(summary["fine_tuning_readiness"], "ready")

    def test_human_corrected_pairs_count_is_accurate(self):
        n = 7
        for _ in range(n):
            self._insert("Produtivo", corrected_category="Improdutivo",
                         subject=str(uuid.uuid4()), content=str(uuid.uuid4()))
        self._insert("Produtivo")  # not corrected
        summary = self.svc.fine_tuning_summary()
        self.assertEqual(summary["human_corrected_pairs"], n)

    def test_training_pairs_total_includes_all(self):
        total = 5
        for _ in range(total):
            self._insert("Produtivo")
        summary = self.svc.fine_tuning_summary()
        self.assertEqual(summary["training_pairs_total"], total)


if __name__ == "__main__":
    unittest.main()
