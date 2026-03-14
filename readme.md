# Email Classification Backend API

## Overview

A Python Flask REST API that classifies emails as **Produtivo** (productive) or **Improdutivo** (unproductive) and generates AI-powered contextual responses in Portuguese.

**Two AI models work together:**
- **Classification (local):** `facebook/bart-large-mnli` — zero-shot classification running inside Docker via Hugging Face Transformers.
- **Response generation (remote):** `Qwen/Qwen2.5-72B-Instruct` — called via the free Hugging Face Inference API.

## Technologies

- **Framework**: Flask 2.3.3
- **ORM**: SQLAlchemy 2.0
- **Database**: MySQL 8.0
- **NLP**: NLTK 3.8
- **Classification AI**: Hugging Face Transformers (`facebook/bart-large-mnli`, local)
- **Response AI**: Hugging Face Inference API (`Qwen/Qwen2.5-72B-Instruct`, remote)
- **Python**: 3.11

## Project Structure

```
python-project/
├── src/
│   ├── __init__.py
│   ├── app.py                         # Flask app factory
│   ├── init_db.py                     # Database initialization
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py                # SQLAlchemy ORM models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── nlp_service.py             # NLTK text preprocessing
│   │   ├── classification_service.py  # Local AI classification
│   │   ├── response_service.py        # Remote AI response generation
│   │   └── training_service.py        # Feedback analytics & training data
│   └── routes/
│       ├── __init__.py
│       └── email_routes.py            # API endpoints
├── templates/
│   └── index.html                     # Web frontend (single-page UI)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── init.sql
└── readme.md
```

## Installation & Setup

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
cd python-project

# 2. Create .env file
cp .env.example .env

# 3. Add your HF API token to .env
#    Get a free token at https://huggingface.co/settings/tokens
#    Edit .env and set: HUGGINGFACE_API_TOKEN=hf_your_token_here

# 4. Start the services
docker compose up -d

# 5. Check logs
docker compose logs -f app

# 6. API available at http://localhost:5000
```

### Option 2: Local Development

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -m nltk.downloader punkt stopwords wordnet omw-1.4

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your MySQL connection details and HF API token

# 5. Run the application
flask run
```

## API Endpoints

### 1. Process Email (Classify & Generate Response)

**POST** `/api/emails/processar`

```json
// Request
{
  "assunto": "Erro no módulo de pagamentos",
  "conteudo": "Prezados, o módulo de pagamentos apresentou erro crítico hoje pela manhã. Precisamos de suporte urgente para resolver."
}

// Response (201)
{
  "email_id": "550e8400-e29b-41d4-a716-446655440000",
  "assunto": "Erro no módulo de pagamentos",
  "conteudo_processado": "erro módulo pagamento ...",
  "classificacao": {
    "categoria": "Produtivo",
    "confianca": 0.92,
    "scores": {
      "Produtivo": 0.92,
      "Improdutivo": 0.08
    },
    "modelo_usado": "facebook/bart-large-mnli"
  },
  "resposta_sugerida": {
    "texto": "Prezado(a),\n\nAgradecemos por nos informar sobre o erro no módulo de pagamentos...",
    "tipo": "ai_generated",
    "modelo_usado": "Qwen/Qwen2.5-72B-Instruct",
    "nivel_personalizacao": "high"
  },
  "timestamp": "2024-03-12T10:30:00"
}
```

### 2. List All Processed Emails

**GET** `/api/emails?page=1&per_page=10&categoria=Produtivo`

### 3. Get Email Details

**GET** `/api/emails/<email_id>`

### 4. Send Feedback

**POST** `/api/emails/<email_id>/feedback`

Accepts two independent types of feedback in a single call (both can be sent together):

```json
// Response quality feedback
{
  "feedback": "helpful",
  "response_id": "550e8400-e29b-41d4-a716-446655440001"
}

// Classification correction (user overrides AI label)
{
  "corrected_category": "Produtivo",
  "feedback_comment": "Optional explanation"
}
```

Feedback values: `helpful`, `not_helpful`  
Category values: `Produtivo`, `Improdutivo`

### 5. Upload Email File

**POST** `/api/emails/upload`

Accepts `multipart/form-data`:
- `file`: `.txt` or `.pdf` (max 5 MB)
- `assunto` *(optional)*: email subject

Returns the same structure as `/api/emails/processar`.

### 6. Training Statistics

**GET** `/api/emails/training/stats`

```json
// Response
{
  "stats": {
    "total_classified": 120,
    "total_corrected": 8,
    "accuracy": 0.9333,
    "corrections_by_direction": {
      "Produtivo → Improdutivo": 5,
      "Improdutivo → Produtivo": 3
    }
  },
  "fine_tuning_readiness": "need_more_data (have 8, need 50)",
  "recommended_approach": "..."
}
```

### 7. Export Training Data

**GET** `/api/emails/training/export`

Query parameters:
- `only_corrected=true` — export only human-corrected examples

Returns a JSONL file (`training_data.jsonl`) ready for Hugging Face `Trainer` or any standard fine-tuning pipeline.

### 8. Health Check

**GET** `/health`

### 9. API Information

**GET** `/`

## Usage Examples

### Classify a Productive Email

```bash
curl -X POST http://localhost:5000/api/emails/processar \
  -H "Content-Type: application/json" \
  -d '{
    "assunto": "Erro crítico no sistema",
    "conteudo": "Identificamos um problema crítico no módulo de processamento de pagamentos. Favor implementar o hotfix imediatamente."
  }'
```

### Classify an Unproductive Email

```bash
curl -X POST http://localhost:5000/api/emails/processar \
  -H "Content-Type: application/json" \
  -d '{
    "assunto": "Boas Festas!",
    "conteudo": "Desejamos a você e sua família um maravilhoso final de ano. Aguardamos você em 2025!"
  }'
```

### List Productive Emails

```bash
curl http://localhost:5000/api/emails?categoria=Produtivo&per_page=20
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
FLASK_APP=src/app.py
FLASK_ENV=development
FLASK_DEBUG=1

MYSQL_HOST=db
MYSQL_PORT=3306
MYSQL_USER=email_user
MYSQL_PASSWORD=email_password
MYSQL_DATABASE=email_classification

SQLALCHEMY_DATABASE_URI=mysql+pymysql://email_user:email_password@db:3306/email_classification

# AI Models
HUGGINGFACE_MODEL_CLASSIFICATION=facebook/bart-large-mnli
HUGGINGFACE_MODEL_RESPONSE=Qwen/Qwen2.5-72B-Instruct

# Free API token — get yours at https://huggingface.co/settings/tokens
HUGGINGFACE_API_TOKEN=hf_your_token_here

LANGUAGE=pt
API_HOST=0.0.0.0
API_PORT=5000
```

## Database Schema

### emails
| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(36) | UUID primary key |
| subject | VARCHAR(255) | Email subject |
| content | LONGTEXT | Email body |
| sender | VARCHAR(255) | Sender |
| received_date | DATETIME | When received |
| created_at | DATETIME | Record creation |

### classifications
| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(36) | UUID primary key |
| email_id | VARCHAR(36) | FK → emails |
| category | VARCHAR(50) | Produtivo / Improdutivo |
| confidence | FLOAT | 0.0–1.0 |
| model_used | VARCHAR(255) | e.g. facebook/bart-large-mnli |
| corrected_category | VARCHAR(50) | Human override (nullable) |
| feedback_comment | TEXT | Optional correction note (nullable) |

### suggested_responses
| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(36) | UUID primary key |
| email_id | VARCHAR(36) | FK → emails |
| category | VARCHAR(50) | Category that triggered the response |
| response_text | LONGTEXT | Generated response |
| model_used | VARCHAR(255) | e.g. Qwen/Qwen2.5-72B-Instruct |
| user_feedback | VARCHAR(50) | helpful / not_helpful |

## Troubleshooting

### Model loading is slow on first request
The BART classification model (~1.6 GB) downloads on first API call. Subsequent requests are fast (150-300ms).

### Response generation returns template instead of AI
Check that `HUGGINGFACE_API_TOKEN` is set in `.env`. Get a free token at https://huggingface.co/settings/tokens.

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "5001:5000"
```

### Database connection error
```bash
docker compose ps          # Check containers are running
docker compose logs db     # Check MySQL logs
```

## Testing

```bash
# Health check
curl http://localhost:5000/health

# Process an email
curl -X POST http://localhost:5000/api/emails/processar \
  -H "Content-Type: application/json" \
  -d '{"assunto": "Teste", "conteudo": "Precisamos de suporte para o sistema de pagamentos."}'

# Run integration tests
python test_api.py
```
