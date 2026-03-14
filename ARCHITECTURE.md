# Email Classification System - Architecture Overview

## 📋 Project Summary

A REST API backend for classifying emails as **Produtivo** (productive) or **Improdutivo** (unproductive) and generating AI-powered contextual responses. Uses two AI models:

- **Classification (local):** `facebook/bart-large-mnli` — zero-shot classification running inside the Docker container via Hugging Face Transformers.
- **Response generation (remote):** `Qwen/Qwen2.5-72B-Instruct` — called via the free Hugging Face Inference API (OpenAI-compatible chat completions endpoint).

The system uses Flask, NLTK for NLP preprocessing, persists data in MySQL, and runs in Docker containers.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT / API CONSUMER                        │
│                  (Web Frontend, Postman, curl)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    HTTP/REST (JSON)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FLASK API SERVER                            │
│               (src/app.py — Port 5000)                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              API Routes (email_routes.py)            │      │
│  │  ├─ POST   /api/emails/processar                    │      │
│  │  ├─ POST   /api/emails/upload                       │      │
│  │  ├─ GET    /api/emails                              │      │
│  │  ├─ GET    /api/emails/<id>                         │      │
│  │  ├─ POST   /api/emails/<id>/feedback                │      │
│  │  ├─ GET    /api/emails/training/stats               │      │
│  │  └─ GET    /api/emails/training/export              │      │
│  └──────────────────┬───────────────────────────────────┘      │
│                     │                                           │
│        ┌────────────┴──────────────┐                           │
│        ▼                           ▼                           │
│  ┌──────────────────┐        ┌─────────────────┐              │
│  │ NLP Service      │        │ Classification  │              │
│  │ (nlp_service.py) │        │ Service         │              │
│  │                  │        │ (classification │              │
│  │ • Preprocessing  │        │  _service.py)   │              │
│  │ • Tokenization   │        │                 │              │
│  │ • Stop words     │        │ • bart-large-   │              │
│  │ • Lemmatization  │        │   mnli (LOCAL)  │              │
│  └──────────────────┘        │ • Zero-shot     │              │
│                              │ • Confidence    │              │
│                              └────────┬────────┘              │
│                                       │                        │
│                              ┌────────▼────────┐              │
│                              │Response Service │              │
│                              │(response_       │              │
│                              │ service.py)     │              │
│                              │                 │              │
│                              │ • Qwen 72B      │              │
│                              │   (HF API)      │              │
│                              │ • Template      │              │
│                              │   fallback      │              │
│                              └────────┬────────┘              │
│                                       │                        │
│                              ┌────────▼────────┐              │
│                              │Training Service │              │
│                              │(training_       │              │
│                              │ service.py)     │              │
│                              │                 │              │
│                              │ • Feedback stats│              │
│                              │ • JSONL export  │              │
│                              │ • Correction    │              │
│                              │   cache lookup  │              │
│                              └────────┬────────┘              │
│                                       │                        │
│        ┌──────────────────────────────┘                        │
│        ▼                                                       │
│  ┌──────────────────────────────────────────────┐             │
│  │    Database Models (database.py)             │             │
│  │  ├─ Email                                    │             │
│  │  ├─ Classification                           │             │
│  │  └─ SuggestedResponse                        │             │
│  │                                              │             │
│  │  SQLAlchemy ORM ◄────────────────► MySQL    │             │
│  └──────────────────┬───────────────────────────┘             │
└─────────────────────┼──────────────────────────────────────────┘
                      │
                      │ SQL Queries (PyMySQL)
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MYSQL 8.0 DATABASE                           │
│              (Port 3306 — Docker Container)                     │
│                                                                 │
│  Tables:                                                        │
│  ├─ emails (id, subject, content, sender, dates)               │
│  ├─ classifications (id, email_id, category, confidence,       │
│  │                   corrected_category, feedback_comment)      │
│  └─ suggested_responses (id, email_id, response, feedback)     │
│                                                                 │
│  Indexes: PRIMARY KEY, FOREIGN KEY, category, confidence       │
└─────────────────────────────────────────────────────────────────┘

                      ▲  External API call (HTTPS)
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│              HUGGING FACE INFERENCE API (Remote)                │
│     https://router.huggingface.co/v1/chat/completions           │
│                                                                 │
│  Model: Qwen/Qwen2.5-72B-Instruct                              │
│  Format: OpenAI-compatible chat completions                     │
│  Auth: Bearer token (HUGGINGFACE_API_TOKEN)                     │
│  Tier: Free                                                     │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### Email Processing Flow

```
1. CLIENT sends JSON request
   {
     "assunto": "...",
     "conteudo": "..."
   }

2. FLASK receives request (POST /api/emails/processar)
   │
   ├─► NLP SERVICE preprocesses text
   │   ├─ Lowercase conversion
   │   ├─ Remove URLs/emails
   │   ├─ Clean special chars
   │   ├─ Tokenization (NLTK)
   │   ├─ Remove stop words
   │   └─ Lemmatization
   │
   ├─► CLASSIFICATION SERVICE (LOCAL AI)
   │   ├─ facebook/bart-large-mnli loaded in-process
   │   ├─ Zero-shot classification
   │   ├─ Calculate confidence scores
   │   └─ Return category (Produtivo/Improdutivo)
   │
   ├─► RESPONSE SERVICE (REMOTE AI)
   │   ├─ Build system + user prompt in Portuguese
   │   ├─ POST to HF Inference API (Qwen 72B)
   │   ├─ Parse chat completion response
   │   ├─ Fallback to template if API unavailable
   │   └─ Return generated response text
   │
   ├─► DATABASE stores results
   │   ├─ Create Email record
   │   ├─ Create Classification record
   │   ├─ Create SuggestedResponse record
   │   └─ Commit to MySQL
   │
   └─► FLASK returns JSON response
       {
         "email_id": "...",
         "classificacao": {...},
         "resposta_sugerida": {...}
       }
```

## 📁 Project Structure

```
python-project/
│
├── Dockerfile                 # Python 3.11-slim image
├── docker-compose.yml         # MySQL + Flask services
├── init.sql                   # Database schema
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore
│
├── src/                       # Application source
│   ├── __init__.py
│   ├── app.py                 # Flask app factory
│   ├── init_db.py             # DB initialization script
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py        # SQLAlchemy ORM models
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── nlp_service.py             # NLTK text preprocessing
│   │   ├── classification_service.py  # Local AI classification
│   │   ├── response_service.py        # Remote AI response generation
│   │   └── training_service.py        # Feedback analytics & training data
│   │
│   └── routes/
│       ├── __init__.py
│       └── email_routes.py    # API endpoints
│
├── templates/
│   └── index.html             # Single-page web frontend
│
├── test_api.py                # API integration tests
├── validate_structure.py      # Project structure validator
│
├── ARCHITECTURE.md            # This file
├── IMPLEMENTATION-SUMMARY.md  # Implementation summary
└── readme.md                  # API documentation
```

## 🔧 Component Details

### 1. Flask Application (src/app.py)
- **Framework**: Flask 2.3.3
- **Database**: SQLAlchemy ORM
- **CORS**: Enabled for cross-origin requests
- **Logging**: Configured for debugging

### 2. Database Layer (src/models/database.py)
**Tables**:

#### emails
```python
id: UUID (Primary Key)
subject: VARCHAR(255)
content: LONGTEXT
sender: VARCHAR(255)
received_date: DATETIME
created_at: DATETIME
```

#### classifications
```python
id: UUID (Primary Key)
email_id: UUID (Foreign Key)
category: VARCHAR(50)  # "Produtivo" or "Improdutivo"
confidence: FLOAT (0.0-1.0)
model_used: VARCHAR(255)
corrected_category: VARCHAR(50)  # Human override (nullable)
feedback_comment: TEXT           # Optional correction note (nullable)
created_at: DATETIME
updated_at: DATETIME
```

#### suggested_responses
```python
id: UUID (Primary Key)
email_id: UUID (Foreign Key)
category: VARCHAR(50)
response_text: LONGTEXT
model_used: VARCHAR(255)
user_feedback: VARCHAR(50)
created_at: DATETIME
updated_at: DATETIME
```

### 3. NLP Service (src/services/nlp_service.py)
**Purpose**: Text preprocessing

**Methods**:
```python
preprocessar_texto(text)        # Main preprocessing
extrair_features(text)          # Feature extraction
```

**Processing Steps**:
1. Convert to lowercase
2. Remove URLs and emails
3. Clean special characters
4. Tokenization (NLTK)
5. Remove stop words
6. Lemmatization

### 4. Classification Service (src/services/classification_service.py)
**Purpose**: Email classification using Hugging Face (runs locally)

**Model**: `facebook/bart-large-mnli` (~1.6 GB, downloaded on first run)
- Zero-shot classification pipeline
- Descriptive category labels for better accuracy
- Runs entirely inside the Docker container

**Methods**:
```python
classificar_email(text, subject)        # Simple classification
classificar_com_detalhes(text, subject)  # Detailed with scores
```

**Output**:
```python
{
    'category': 'Produtivo',
    'confidence': 0.92,
    'scores': {'Produtivo': 0.92, 'Improdutivo': 0.08},
    'model_used': 'facebook/bart-large-mnli'
}
```

### 5. Response Service (src/services/response_service.py)
**Purpose**: Generate contextual email responses in Portuguese

**Approach**: Hugging Face Inference API (remote) with template fallback

**How it works**:
1. Builds a system prompt (professional financial-sector assistant) and a user prompt containing the email
2. Calls `POST https://router.huggingface.co/v1/chat/completions` with the `Qwen/Qwen2.5-72B-Instruct` model
3. Parses the chat completion response
4. If the API is unavailable or no token is configured, falls back to template-based responses

**Auth**: Requires `HUGGINGFACE_API_TOKEN` environment variable (free tier)

### 6. Training Service (src/services/training_service.py)
**Purpose**: Manage feedback-derived training data and model accuracy analytics

**Methods**:
```python
get_feedback_stats()                        # Accuracy metrics from corrections
get_training_pairs(only_corrected=False)    # Labeled pairs for fine-tuning
export_as_jsonl(only_corrected=False)       # JSONL string for Trainer
find_correction_for_email(subject, content) # Feedback cache lookup
fine_tuning_summary()                       # Readiness report
```

**Feedback cache**: When processing a new email, the route layer checks if an identical email was previously corrected by a human. If found, the human label is applied directly (confidence=1.0, model=`human_correction`).

**Fine-tuning threshold**: 50 human-corrected examples triggers `ready` readiness status.

### 7. API Routes (src/routes/email_routes.py)
**Endpoints**:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/emails/processar` | Classify email & generate response |
| POST | `/api/emails/upload` | Process email from .txt/.pdf file |
| GET | `/api/emails` | List all processed emails |
| GET | `/api/emails/<id>` | Get email details |
| POST | `/api/emails/<id>/feedback` | Submit response feedback and/or classification correction |
| GET | `/api/emails/training/stats` | Model accuracy metrics |
| GET | `/api/emails/training/export` | Export labeled JSONL training data |
| GET | `/health` | Health check |
| GET | `/` | API info |

## 🚀 Deployment

### Docker Deployment (Recommended)
```
1. Set HUGGINGFACE_API_TOKEN in .env
2. docker compose up -d
3. MySQL starts + health check
4. init_db.py creates/verifies tables
5. Flask starts on port 5000
6. BART classification model downloads on first request (~1.6 GB)
7. Response generation calls remote HF API (no local download)
```

## 📊 Database Relationships

```
emails (1) ──────────► (1) classifications
        └─ Referenced by email_id (Foreign Key)

emails (1) ──────────► (many) suggested_responses
        └─ Referenced by email_id (Foreign Key)
```

## 📈 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| First request | 30-60s | BART model download + loading |
| Classification | 150-300ms | Local inference (per email) |
| Response generation | 2-5s | Remote API call (Qwen 72B) |
| Database insert | 50-100ms | SQL operations |
| Full process (warm) | 3-6s | After model loaded |

## 🔄 Scalability Improvements

1. **Caching**: Cache model in memory
2. **Batch Processing**: Process multiple emails at once
3. **Async Tasks**: Use Celery for async processing
4. **Load Balancing**: Multiple Flask instances
5. **Database Optimization**: Connection pooling

## 🧪 Testing Strategy

### Unit Tests
- Individual service methods
- NLP preprocessing
- Response generation logic

### Integration Tests
- API endpoints
- Database operations
- Service integration

### E2E Tests
- Full email processing flow
- API response validation
- Database persistence

## 📝 Future Enhancements

1. **Generative Models**: Implement generative response creation
2. **Multi-language**: Support Portuguese natively
3. **Model Fine-tuning**: Train on custom data
4. **Batch Processing**: Endpoint for multiple emails
5. **File Upload**: Support PDF and TXT files
6. **Analytics Dashboard**: Metrics and statistics
7. **Email Integration**: Connect to Gmail/Outlook APIs
8. **A/B Testing**: Test different response templates

## 🔗 Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 2.3.3 | Web framework |
| SQLAlchemy | 2.0.21 | ORM |
| Flask-SQLAlchemy | 3.0.5 | Flask + SQL |
| PyMySQL | 1.1.0 | MySQL driver |
| torch | 2.0.1 | PyTorch (for transformers) |
| transformers | 4.33.2 | Hugging Face models |
| nltk | 3.8.1 | NLP toolkit |
| Flask-CORS | 4.0.0 | CORS support |
| python-dotenv | 1.0.0 | Environment variables |
| gunicorn | 21.2.0 | Production server |

---

**Created**: March 12, 2026
**Version**: 1.0.0
**Status**: Ready for deployment

For detailed API documentation, see [README-BACKEND.md](README-BACKEND.md)
