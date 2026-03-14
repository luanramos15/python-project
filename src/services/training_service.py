"""
Training service: converts user feedback into labeled training data and
provides statistics about model accuracy so the classifier can be
periodically fine-tuned or its prompt-engineering improved.

Workflow
--------
1. User submits an email → model classifies it.
2. If the user disagrees, they call POST /api/emails/<id>/feedback with
   { "corrected_category": "Produtivo" }.  The correction is stored in
   classifications.corrected_category.
3. POST /api/training/export returns all verified pairs (subject, body,
   correct_label) ready for fine-tuning.
4. The optional apply_feedback_corrections() helper lets the route layer
   override a freshly computed classification when a correction already
   exists in the database for an identical or near-identical email.
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for managing training data derived from user feedback."""

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_feedback_stats(self):
        """
        Return accuracy metrics derived from stored corrections.

        Returns
        -------
        dict with keys:
            total_classified    – total emails processed
            total_corrected     – emails where users overrode the AI
            accuracy            – fraction the AI got right (0–1)
            corrections_by_direction – how often each type of flip occurred
        """
        # Imported here to avoid a circular import at module level
        from src.models.database import Classification

        all_classifications = Classification.query.all()
        total = len(all_classifications)

        corrected = [c for c in all_classifications if c.corrected_category is not None]
        n_corrected = len(corrected)

        accuracy = round(1.0 - (n_corrected / total), 4) if total > 0 else None

        # Count direction of corrections (e.g. Produtivo→Improdutivo)
        directions: dict[str, int] = {}
        for c in corrected:
            key = f"{c.category} → {c.corrected_category}"
            directions[key] = directions.get(key, 0) + 1

        return {
            "total_classified": total,
            "total_corrected": n_corrected,
            "accuracy": accuracy,
            "corrections_by_direction": directions,
        }

    # ------------------------------------------------------------------
    # Training data export
    # ------------------------------------------------------------------

    def get_training_pairs(self, only_corrected: bool = False):
        """
        Build a list of labeled text pairs suitable for fine-tuning.

        Each item is:
            {
                "email_id": str,
                "subject": str,
                "content": str,
                "label": str,          # the *correct* label
                "original_label": str, # what the model predicted
                "was_corrected": bool,
                "source": "human_correction" | "model_confirmed"
            }

        Parameters
        ----------
        only_corrected : bool
            If True, return only the examples where the model was wrong.
            Useful for targeted fine-tuning on hard cases.
        """
        from src.models.database import Email, Classification

        results = []

        query = Classification.query
        if only_corrected:
            query = query.filter(Classification.corrected_category.isnot(None))

        for clf in query.all():
            email: Email = clf.email
            if email is None:
                continue

            was_corrected = clf.corrected_category is not None
            correct_label = clf.corrected_category if was_corrected else clf.category

            results.append(
                {
                    "email_id": clf.email_id,
                    "subject": email.subject,
                    "content": email.content,
                    "label": correct_label,
                    "original_label": clf.category,
                    "confidence": clf.confidence,
                    "feedback_comment": clf.feedback_comment,
                    "was_corrected": was_corrected,
                    "source": "human_correction" if was_corrected else "model_confirmed",
                }
            )

        return results

    def export_as_jsonl(self, only_corrected: bool = False) -> str:
        """Return training pairs serialized as newline-delimited JSON."""
        pairs = self.get_training_pairs(only_corrected=only_corrected)
        lines = [json.dumps(pair, ensure_ascii=False) for pair in pairs]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Feedback-aware override (lightweight "learning from corrections")
    # ------------------------------------------------------------------

    def find_correction_for_email(self, subject: str, content: str):
        """
        Check whether the database already contains a human correction for an
        email with this *exact* subject and content.  If so, return the
        corrected label and the confidence to use; otherwise return None.

        This is intentionally conservative: only exact-match overrides are
        applied so that accidental feedback cannot silently corrupt unrelated
        emails.
        """
        from src.models.database import Email, Classification

        # Normalise for comparison
        norm_subject = (subject or "").strip().lower()
        norm_content = (content or "").strip().lower()

        # Fetch corrections that have been explicitly set by humans
        corrected = (
            Classification.query.filter(
                Classification.corrected_category.isnot(None)
            )
            .join(Email)
            .all()
        )

        for clf in corrected:
            email: Email = clf.email
            if (
                email.subject.strip().lower() == norm_subject
                and email.content.strip().lower() == norm_content
            ):
                logger.info(
                    "Feedback cache hit: overriding model output with human "
                    f"correction '{clf.corrected_category}' for email_id={clf.email_id}"
                )
                return {
                    "category": clf.corrected_category,
                    "confidence": 1.0,  # human label is treated as ground truth
                    "source": "human_correction",
                }

        return None

    # ------------------------------------------------------------------
    # Fine-tuning guidance
    # ------------------------------------------------------------------

    def fine_tuning_summary(self) -> dict:
        """
        Return a structured summary of what fine-tuning would require,
        together with a readiness assessment based on the volume of
        available labeled data.
        """
        stats = self.get_feedback_stats()
        pairs = self.get_training_pairs(only_corrected=False)
        corrected_pairs = [p for p in pairs if p["was_corrected"]]

        MIN_SAMPLES_FOR_FINE_TUNE = 50

        readiness = (
            "ready"
            if len(corrected_pairs) >= MIN_SAMPLES_FOR_FINE_TUNE
            else f"need_more_data (have {len(corrected_pairs)}, "
            f"need {MIN_SAMPLES_FOR_FINE_TUNE})"
        )

        return {
            "stats": stats,
            "training_pairs_total": len(pairs),
            "human_corrected_pairs": len(corrected_pairs),
            "fine_tuning_readiness": readiness,
            "recommended_approach": (
                "fine-tune facebook/bart-large-mnli on the exported JSONL "
                "using HuggingFace Trainer, then replace the model_name in "
                "ClassificationService with the new checkpoint."
            ),
            "export_endpoint": "POST /api/training/export",
            "generated_at": datetime.utcnow().isoformat(),
        }
