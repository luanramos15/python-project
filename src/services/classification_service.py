from transformers import pipeline
import logging
import os

logger = logging.getLogger(__name__)


class ClassificationService:
    """
    Service for classifying emails using Hugging Face Transformers.
    Uses zero-shot classification to categorize emails as "Produtivo" or "Improdutivo".
    """

    def __init__(self, model_name=None):
        """
        Initialize the classification service.
        
        Args:
            model_name: Name of the model to use (default: facebook/bart-large-mnli for better zero-shot)
        """
        self.model_name = model_name or os.getenv('HUGGINGFACE_MODEL_CLASSIFICATION', 'facebook/bart-large-mnli')
        
        # Use more descriptive labels for better zero-shot classification
        self.category_descriptions = {
            'Produtivo': 'work-related, professional, business communication, project updates, technical issues, requests for information, official correspondence',
            'Improdutivo': 'personal, social, casual conversation, greetings, party invitations, non-work related topics, entertainment, leisure activities'
        }
        self.categories = list(self.category_descriptions.keys())
        
        try:
            logger.info(f"Loading classification model: {self.model_name}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1  # Use CPU; set to 0 for GPU if available
            )
            logger.info("Classification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            raise

    def classificar_email(self, text, subject=""):
        """
        Classify an email as either 'Produtivo' or 'Improdutivo'.
        
        Args:
            text: Email content/body
            subject: Email subject (optional, will be combined with content)
            
        Returns:
            dict: Classification result containing:
                - category: The predicted category
                - confidence: Confidence score (0.0 to 1.0)
                - scores: Dictionary with scores for each category
                - model_used: Name of the model used
        """
        if not text or not text.strip():
            return {
                'category': 'Improdutivo',
                'confidence': 0.0,
                'scores': {'Produtivo': 0.0, 'Improdutivo': 1.0},
                'model_used': self.model_name,
                'error': 'Empty text provided'
            }

        try:
            # Combine subject and text for better classification
            combined_text = f"{subject} {text}".strip()
            
            # Limit text length for performance
            combined_text = combined_text[:512]

            # Use descriptive labels for better zero-shot classification
            candidate_labels = list(self.category_descriptions.values())
            
            # Perform zero-shot classification
            result = self.classifier(combined_text, candidate_labels, multi_class=False)

            # Extract and structure results
            labels = result['labels']
            scores = result['scores']

            # Map back to our category names
            category_scores = {}
            for desc_label, score in zip(labels, scores):
                # Find which category this description belongs to
                for category, description in self.category_descriptions.items():
                    if desc_label == description:
                        category_scores[category] = score
                        break

            # Get the top category
            top_category = max(category_scores, key=category_scores.get)

            return {
                'category': top_category,
                'confidence': float(category_scores[top_category]),
                'scores': category_scores,
                'model_used': self.model_name,
            }

        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return {
                'category': None,
                'confidence': 0.0,
                'scores': {},
                'model_used': self.model_name,
                'error': str(e)
            }

    def classificar_com_detalhes(self, text, subject=""):
        """
        Classify an email and return additional details for analysis.
        
        Args:
            text: Email content/body
            subject: Email subject
            
        Returns:
            dict: Detailed classification result
        """
        classification = self.classificar_email(text, subject)
        
        # Add text length and content type indicators
        classification['text_length'] = len(text)
        classification['has_attachment_keywords'] = any(
            keyword in text.lower() for keyword in 
            ['anexo', 'attachment', 'arquivo', 'file', 'documento', 'document']
        )
        classification['has_action_keywords'] = any(
            keyword in text.lower() for keyword in 
            ['precisa', 'preciso', 'urgente', 'favor', 'solicitação', 'ajuda', 
             'help', 'issue', 'problema', 'problem', 'request', 'solici']
        )
        
        return classification


# Create a singleton instance
classification_service = ClassificationService()
