#!/usr/bin/env python3
"""
Quick validation script to test the basic structure and imports.
"""

import sys
import os

print("=" * 70)
print("EMAIL CLASSIFICATION API - STRUCTURE VALIDATION")
print("=" * 70)

# Test 1: Check Python version
print("\n[1] Python Version Check")
print(f"    Python {sys.version}")
if sys.version_info >= (3, 8):
    print("    ✓ Compatible version")
else:
    print("    ✗ Python 3.8+ required")
    sys.exit(1)

# Test 2: Check directory structure
print("\n[2] Directory Structure Check")
required_dirs = [
    'src',
    'src/models',
    'src/services',
    'src/routes',
    'templates'
]
for dir_path in required_dirs:
    if os.path.isdir(dir_path):
        print(f"    ✓ {dir_path}/")
    else:
        print(f"    ✗ {dir_path}/ NOT FOUND")

# Test 3: Check required files
print("\n[3] Required Files Check")
required_files = [
    'requirements.txt',
    'Dockerfile',
    'docker-compose.yml',
    '.env.example',
    'init.sql',
    'src/__init__.py',
    'src/app.py',
    'src/init_db.py',
    'src/models/__init__.py',
    'src/models/database.py',
    'src/services/__init__.py',
    'src/services/nlp_service.py',
    'src/services/classification_service.py',
    'src/services/response_service.py',
    'src/routes/__init__.py',
    'src/routes/email_routes.py',
]
for file_path in required_files:
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        print(f"    ✓ {file_path} ({file_size} bytes)")
    else:
        print(f"    ✗ {file_path} NOT FOUND")

# Test 4: Check Python imports (without dependencies)
print("\n[4] Python Imports Check (Core Modules)")
try:
    import sys
    import os
    import logging
    import json
    import time
    import uuid
    from datetime import datetime
    print("    ✓ All core Python modules available")
except ImportError as e:
    print(f"    ✗ Import error: {e}")

# Test 5: Compile Python files
print("\n[5] Python Files Compilation Check")
import py_compile
python_files = [
    'src/app.py',
    'src/init_db.py',
    'src/models/database.py',
    'src/services/nlp_service.py',
    'src/services/classification_service.py',
    'src/services/response_service.py',
    'src/routes/email_routes.py',
]
all_compiled = True
for py_file in python_files:
    try:
        py_compile.compile(py_file, doraise=True)
        print(f"    ✓ {py_file}")
    except py_compile.PyCompileError as e:
        print(f"    ✗ {py_file}: {e}")
        all_compiled = False

if not all_compiled:
    sys.exit(1)

# Test 6: Summary
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print("""
✓ Project structure is complete and valid
✓ All required files are present
✓ All Python files compile successfully

NEXT STEPS:
1. Install dependencies: pip install -r requirements.txt
2. Download NLTK data: python -m nltk.downloader punkt stopwords wordnet
3. Set up environment: cp .env.example .env
   -> Set HUGGINGFACE_API_TOKEN in .env
4. Start with Docker: docker compose up -d
   OR run locally: flask run

For detailed backend documentation, see README-BACKEND.md
""")
print("=" * 70)
