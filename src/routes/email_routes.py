from flask import Blueprint, request, jsonify
import logging
import io
from src.models.database import db, Email, Classification, SuggestedResponse
from src.services import nlp_service, classification_service, response_service

logger = logging.getLogger(__name__)

# Create blueprint
email_bp = Blueprint('emails', __name__, url_prefix='/api/emails')


@email_bp.route('/processar', methods=['POST'])
def processar_email():
    """
    Process a single email: classify it and generate a suggested response.
    
    Expected JSON payload:
    {
        "assunto": "Email subject",
        "conteudo": "Email body/content"
    }
    
    Returns:
        dict: Processed email with classification and suggested response
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        assunto = data.get('assunto', '').strip()
        conteudo = data.get('conteudo', '').strip()
        
        if not conteudo:
            return jsonify({'error': 'Content (conteudo) is required'}), 400
        
        # Preprocess text with NLP
        logger.info(f"Preprocessing email with subject: {assunto}")
        processed_text, tokens = nlp_service.preprocessar_texto(conteudo)
        
        # Classify email
        logger.info("Classifying email...")
        classification_result = classification_service.classificar_email(conteudo, assunto)
        
        if classification_result.get('error'):
            return jsonify({
                'error': f"Classification error: {classification_result['error']}"
            }), 500
        
        categoria = classification_result['category']
        confianca = classification_result['confidence']
        
        # Generate response
        logger.info(f"Generating response for category: {categoria}")
        response_result = response_service.gerar_resposta(conteudo, assunto, categoria)
        
        # Store in database
        email = Email(
            subject=assunto,
            content=conteudo
        )
        db.session.add(email)
        db.session.flush()
        
        classification = Classification(
            email_id=email.id,
            category=categoria,
            confidence=confianca,
            model_used=classification_result.get('model_used')
        )
        db.session.add(classification)
        
        suggested_response = SuggestedResponse(
            email_id=email.id,
            category=categoria,
            response_text=response_result['response_text'],
            model_used=response_result.get('model_used')
        )
        db.session.add(suggested_response)
        
        db.session.commit()
        
        return jsonify({
            'email_id': email.id,
            'assunto': assunto,
            'conteudo_processado': processed_text,
            'classificacao': {
                'categoria': categoria,
                'confianca': confianca,
                'scores': classification_result.get('scores', {}),
                'modelo_usado': classification_result.get('model_used')
            },
            'resposta_sugerida': {
                'texto': response_result['response_text'],
                'tipo': response_result.get('subtype'),
                'modelo_usado': response_result.get('model_used'),
                'nivel_personalizacao': response_result.get('personalization_level')
            },
            'timestamp': email.created_at.isoformat()
        }), 201
        
    except Exception as e:
        logger.error(f"Error processing email: {e}")
        db.session.rollback()
        return jsonify({'error': f"Server error: {str(e)}"}), 500


@email_bp.route('', methods=['GET'])
def listar_emails():
    """
    List all processed emails with their classifications and suggested responses.
    
    Query parameters:
        - page: Page number (default: 1)
        - per_page: Items per page (default: 10)
        - categoria: Filter by category (Produtivo/Improdutivo)
    
    Returns:
        dict: List of emails with pagination
    """
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        categoria = request.args.get('categoria', None, type=str)
        
        # Build query
        query = Email.query
        
        if categoria:
            query = query.join(Classification).filter(Classification.category == categoria)
        
        # Paginate
        paginated = query.paginate(page=page, per_page=per_page, error_out=False)
        
        emails_data = []
        for email in paginated.items:
            email_dict = email.to_dict()
            if email.classifications:
                email_dict['classificacao'] = email.classifications.to_dict()
            if email.suggested_responses:
                email_dict['respostas'] = [r.to_dict() for r in email.suggested_responses]
            emails_data.append(email_dict)
        
        return jsonify({
            'emails': emails_data,
            'total': paginated.total,
            'pages': paginated.pages,
            'current_page': page,
            'per_page': per_page
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing emails: {e}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


@email_bp.route('/<email_id>', methods=['GET'])
def obter_email(email_id):
    """
    Get details of a specific processed email.
    
    Args:
        email_id: UUID of the email
    
    Returns:
        dict: Email with classification and responses
    """
    try:
        email = Email.query.get(email_id)
        
        if not email:
            return jsonify({'error': f"Email {email_id} not found"}), 404
        
        email_dict = email.to_dict()
        if email.classifications:
            email_dict['classificacao'] = email.classifications.to_dict()
        if email.suggested_responses:
            email_dict['respostas'] = [r.to_dict() for r in email.suggested_responses]
        
        return jsonify(email_dict), 200
        
    except Exception as e:
        logger.error(f"Error retrieving email: {e}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500


@email_bp.route('/<email_id>/feedback', methods=['POST'])
def enviar_feedback(email_id):
    """
    Submit feedback about the generated classification or response.
    
    Expected JSON payload:
    {
        "feedback": "helpful" | "not_helpful",
        "response_id": "response_id (optional)"
    }
    
    Returns:
        dict: Confirmation of feedback submission
    """
    try:
        email = Email.query.get(email_id)
        
        if not email:
            return jsonify({'error': f"Email {email_id} not found"}), 404
        
        data = request.get_json()
        feedback = data.get('feedback', '').strip()
        response_id = data.get('response_id', None)
        
        if not feedback:
            return jsonify({'error': 'Feedback is required'}), 400
        
        # Update response feedback if specified
        if response_id:
            response = SuggestedResponse.query.get(response_id)
            if response and response.email_id == email_id:
                response.user_feedback = feedback
                db.session.commit()
        
        return jsonify({
            'email_id': email_id,
            'feedback': feedback,
            'message': 'Feedback registered successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error registering feedback: {e}")
        db.session.rollback()
        return jsonify({'error': f"Server error: {str(e)}"}), 500


ALLOWED_EXTENSIONS = {'.txt', '.pdf'}


@email_bp.route('/upload', methods=['POST'])
def upload_email():
    """
    Process an email from a .txt or .pdf file upload.

    Expects multipart/form-data with:
        - file: a .txt or .pdf file
        - assunto (optional): email subject

    Returns the same structure as /processar.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

        import os
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Formato não suportado. Use .txt ou .pdf'}), 400

        # Extract text from file
        if ext == '.txt':
            raw = file.read()
            # Try UTF-8 first, then latin-1 as fallback
            try:
                conteudo = raw.decode('utf-8')
            except UnicodeDecodeError:
                conteudo = raw.decode('latin-1')
        else:  # .pdf
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file.read()))
            pages = [page.extract_text() or '' for page in reader.pages]
            conteudo = '\n'.join(pages)

        conteudo = conteudo.strip()
        if not conteudo:
            return jsonify({'error': 'O arquivo está vazio ou não foi possível extrair texto'}), 400

        assunto = request.form.get('assunto', file.filename).strip()

        # Reuse the same processing pipeline as /processar
        logger.info(f"Processing uploaded file: {file.filename}")
        processed_text, tokens = nlp_service.preprocessar_texto(conteudo)

        classification_result = classification_service.classificar_email(conteudo, assunto)
        if classification_result.get('error'):
            return jsonify({'error': f"Classification error: {classification_result['error']}"}), 500

        categoria = classification_result['category']
        confianca = classification_result['confidence']

        response_result = response_service.gerar_resposta(conteudo, assunto, categoria)

        email = Email(subject=assunto, content=conteudo)
        db.session.add(email)
        db.session.flush()

        classification = Classification(
            email_id=email.id,
            category=categoria,
            confidence=confianca,
            model_used=classification_result.get('model_used')
        )
        db.session.add(classification)

        suggested_response = SuggestedResponse(
            email_id=email.id,
            category=categoria,
            response_text=response_result['response_text'],
            model_used=response_result.get('model_used')
        )
        db.session.add(suggested_response)
        db.session.commit()

        return jsonify({
            'email_id': email.id,
            'assunto': assunto,
            'conteudo_processado': processed_text,
            'classificacao': {
                'categoria': categoria,
                'confianca': confianca,
                'scores': classification_result.get('scores', {}),
                'modelo_usado': classification_result.get('model_used')
            },
            'resposta_sugerida': {
                'texto': response_result['response_text'],
                'tipo': response_result.get('subtype'),
                'modelo_usado': response_result.get('model_used'),
                'nivel_personalizacao': response_result.get('personalization_level')
            },
            'timestamp': email.created_at.isoformat()
        }), 201

    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        db.session.rollback()
        return jsonify({'error': f"Erro no servidor: {str(e)}"}), 500
