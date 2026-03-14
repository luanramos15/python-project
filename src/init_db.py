"""
Database verification script.
This script ensures the database is ready and tables exist before the application starts.
Tables are created by init.sql, this script only verifies they exist.
"""

import os
import sys
import time
import logging
import mysql.connector
from mysql.connector import Error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def wait_for_database(host, port, user, password, database, max_retries=30, wait_time=2):
    """
    Wait for the MySQL database to be ready.
    
    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        database: Database name
        max_retries: Maximum number of retry attempts
        wait_time: Time to wait between retries (seconds)
        
    Returns:
        bool: True if database is ready, False otherwise
    """
    retries = 0
    
    while retries < max_retries:
        try:
            logger.info(f"Attempting to connect to database at {host}:{port}...")
            connection = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            
            if connection.is_connected():
                logger.info("✓ Database connection successful!")
                connection.close()
                return True
                
        except Error as err:
            retries += 1
            logger.warning(f"Connection attempt {retries}/{max_retries} failed: {err}")
            
            if retries < max_retries:
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    logger.error(f"✗ Failed to connect to database after {max_retries} attempts")
    return False


def initialize_database():
    """
    Verify that the database and required tables exist.
    Tables are created by init.sql, so we only verify here.
    """
    from src.models.database import db
    from src.app import create_app
    
    try:
        logger.info("Creating Flask application context...")
        app = create_app()
        
        with app.app_context():
            logger.info("Verifying database tables...")
            
            # Verify tables were created by init.sql
            inspector_result = db.inspect(db.engine)
            tables = inspector_result.get_table_names()
            
            logger.info(f"Tables in database: {', '.join(tables)}")
            
            required_tables = {'emails', 'classifications', 'suggested_responses'}
            existing_tables = set(tables)
            
            if required_tables.issubset(existing_tables):
                logger.info("✓ All required tables are present")
                return True
            else:
                missing = required_tables - existing_tables
                logger.error(f"✗ Missing tables: {missing}")
                logger.error("Tables should have been created by init.sql")
                return False
                
    except Exception as e:
        logger.error(f"✗ Error during database initialization: {e}")
        return False


if __name__ == '__main__':
    use_sqlite = os.getenv('USE_SQLITE', '').lower() in ('1', 'true', 'yes')

    if use_sqlite:
        logger.info("=" * 60)
        logger.info("Database Initialization Script (SQLite mode)")
        logger.info("=" * 60)
        logger.info("Skipping MySQL wait — using SQLite")
        logger.info("\nInitializing database via SQLAlchemy...")
        if not initialize_database():
            logger.error("Database initialization failed. Exiting.")
            sys.exit(1)
        logger.info("\n" + "=" * 60)
        logger.info("✓ SQLite database ready!")
        logger.info("=" * 60)
        sys.exit(0)

    # MySQL mode
    # Get database configuration from environment variables
    db_host = os.getenv('MYSQL_HOST', 'db')
    db_port = int(os.getenv('MYSQL_PORT', 3306))
    db_user = os.getenv('MYSQL_USER', 'email_user')
    db_password = os.getenv('MYSQL_PASSWORD', 'email_password')
    db_name = os.getenv('MYSQL_DATABASE', 'email_classification')
    
    logger.info("=" * 60)
    logger.info("Database Initialization Script")
    logger.info("=" * 60)
    logger.info(f"Database Host: {db_host}")
    logger.info(f"Database Port: {db_port}")
    logger.info(f"Database Name: {db_name}")
    logger.info("=" * 60)
    
    # Wait for database to be ready
    logger.info("\nStep 1: Waiting for database to be ready...")
    if not wait_for_database(db_host, db_port, db_user, db_password, db_name):
        logger.error("Database is not ready. Exiting.")
        sys.exit(1)
    
    # Verify database
    logger.info("\nStep 2: Verifying database...")
    if not initialize_database():
        logger.error("Database verification failed. Exiting.")
        sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Database verification completed successfully!")
    logger.info("=" * 60)
    sys.exit(0)
