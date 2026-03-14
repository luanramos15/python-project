import os
import logging
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy import text
from src.models.database import db
from src.routes import email_bp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """
    Application factory function to create and configure the Flask app.
    
    Returns:
        Flask: Configured Flask application
    """
    
    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))
    
    # Configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'SQLALCHEMY_DATABASE_URI',
        'mysql+pymysql://root:seu_password_seguro@localhost:3306/email_classification'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB upload limit
    
    logger.info(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
    
    # Initialize extensions
    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprints
    app.register_blueprint(email_bp)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint for monitoring."""
        try:
            # Test database connection
            db.session.execute(text('SELECT 1'))
            return jsonify({
                'status': 'healthy',
                'message': 'Application is running',
                'database': 'connected'
            }), 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'message': f"Database connection failed: {str(e)}"
            }), 500
    
    # Root endpoint — serves the frontend
    @app.route('/', methods=['GET'])
    def index():
        """Serve the single-page frontend."""
        return render_template('index.html')

    # API info endpoint (moved from /)
    @app.route('/api/info', methods=['GET'])
    def api_info():
        """API information and available endpoints."""
        return jsonify({
            'application': 'Email Classification API',
            'version': '1.0.0',
            'description': 'API for classifying emails as Produtivo or Improdutivo using NLP and AI',
            'endpoints': {
                'health': '/health',
                'process_email': 'POST /api/emails/processar',
                'upload_email': 'POST /api/emails/upload',
                'list_emails': 'GET /api/emails',
                'get_email': 'GET /api/emails/<email_id>',
                'send_feedback': 'POST /api/emails/<email_id>/feedback'
            }
        }), 200
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    # Create tables
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    logger.info("Flask application initialized successfully")
    
    return app


# Create the app instance
app = create_app()


if __name__ == '__main__':
    app.run(
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', False)
    )
