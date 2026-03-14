from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()


class Email(db.Model):
    """
    Model representing an email to be classified.
    """
    __tablename__ = 'emails'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    subject = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String(255), nullable=True)
    received_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # Relations
    classifications = db.relationship('Classification', backref='email', uselist=False, cascade='all, delete-orphan')
    suggested_responses = db.relationship('SuggestedResponse', backref='email', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Email {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'subject': self.subject,
            'content': self.content,
            'sender': self.sender,
            'received_date': self.received_date.isoformat() if self.received_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class Classification(db.Model):
    """
    Model representing the classification of an email.
    """
    __tablename__ = 'classifications'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email_id = db.Column(db.String(36), db.ForeignKey('emails.id'), nullable=False, unique=True)
    category = db.Column(db.String(50), nullable=False)  # "Produtivo" or "Improdutivo"
    confidence = db.Column(db.Float, nullable=False)  # 0.0 to 1.0
    model_used = db.Column(db.String(255), nullable=True)  # Name of the model used
    # Feedback fields: corrected_category is set when a user overrides the AI's prediction
    corrected_category = db.Column(db.String(50), nullable=True)
    feedback_comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Classification {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'email_id': self.email_id,
            'category': self.category,
            'confidence': self.confidence,
            'model_used': self.model_used,
            'corrected_category': self.corrected_category,
            'feedback_comment': self.feedback_comment,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class SuggestedResponse(db.Model):
    """
    Model representing a suggested response for an email.
    """
    __tablename__ = 'suggested_responses'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email_id = db.Column(db.String(36), db.ForeignKey('emails.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # Category for which the response was generated
    response_text = db.Column(db.Text, nullable=False)
    model_used = db.Column(db.String(255), nullable=True)  # Name of the model used
    user_feedback = db.Column(db.String(50), nullable=True)  # "helpful", "not_helpful", etc.
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<SuggestedResponse {self.id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'email_id': self.email_id,
            'category': self.category,
            'response_text': self.response_text,
            'model_used': self.model_used,
            'user_feedback': self.user_feedback,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

