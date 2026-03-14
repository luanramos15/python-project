-- Database initialization script for email classification system

-- Create database if not exists (this will be created by Docker, but just in case)
CREATE DATABASE IF NOT EXISTS email_classification;

-- Use the database
USE email_classification;

-- Create emails table
CREATE TABLE IF NOT EXISTS emails (
    id VARCHAR(36) PRIMARY KEY,
    subject VARCHAR(255) NOT NULL,
    content LONGTEXT NOT NULL,
    sender VARCHAR(255),
    received_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_subject (subject),
    INDEX idx_received_date (received_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create classifications table
CREATE TABLE IF NOT EXISTS classifications (
    id VARCHAR(36) PRIMARY KEY,
    email_id VARCHAR(36) NOT NULL,
    category VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    model_used VARCHAR(255),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_email_id (email_id),
    FOREIGN KEY (email_id) REFERENCES emails(id) ON DELETE CASCADE,
    INDEX idx_category (category),
    INDEX idx_confidence (confidence)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create suggested_responses table
CREATE TABLE IF NOT EXISTS suggested_responses (
    id VARCHAR(36) PRIMARY KEY,
    email_id VARCHAR(36) NOT NULL,
    category VARCHAR(50) NOT NULL,
    response_text LONGTEXT NOT NULL,
    model_used VARCHAR(255),
    user_feedback VARCHAR(50),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (email_id) REFERENCES emails(id) ON DELETE CASCADE,
    INDEX idx_email_id (email_id),
    INDEX idx_category (category),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create indexes for better query performance
ALTER TABLE emails ADD FULLTEXT INDEX idx_fulltext_emails (subject, content);

-- Create user and grant privileges
CREATE USER IF NOT EXISTS 'email_user'@'%' IDENTIFIED BY 'email_password';
GRANT ALL PRIVILEGES ON email_classification.* TO 'email_user'@'%';
FLUSH PRIVILEGES;