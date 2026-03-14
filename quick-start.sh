#!/bin/bash
# Quick Start Script for Email Classification API
# Run: bash quick-start.sh

set -e

echo "=========================================="
echo "Email Classification API - Quick Start"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}[!] Docker is not installed.${NC}"
    echo "    Please install Docker Desktop or Docker Engine."
    echo "    https://www.docker.com/"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo -e "${YELLOW}[!] 'docker compose' plugin is not available.${NC}"
    echo "    Please install Docker Compose V2 (comes with Docker Desktop)." 
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${BLUE}[1/5] Setting up environment variables...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo -e "${YELLOW}    ⚠ Edit .env and set HUGGINGFACE_API_TOKEN${NC}"
    echo "    Get a free token at https://huggingface.co/settings/tokens"
else
    echo -e "${BLUE}[1/5] Using existing .env file${NC}"
fi

echo ""
echo -e "${BLUE}[2/5] Checking HF API token...${NC}"
if grep -q 'HUGGINGFACE_API_TOKEN=hf_' .env 2>/dev/null; then
    echo -e "${GREEN}✓ HF API token is set${NC}"
else
    echo -e "${YELLOW}⚠ HUGGINGFACE_API_TOKEN not set in .env — responses will use template fallback${NC}"
    echo "    Get a free token at https://huggingface.co/settings/tokens"
fi

echo ""
echo -e "${BLUE}[3/5] Validating project structure...${NC}"
python3 validate_structure.py > /dev/null 2>&1 && echo -e "${GREEN}✓ Structure is valid${NC}" || echo -e "${YELLOW}[!] Structure validation issues${NC}"

echo ""
echo -e "${BLUE}[4/5] Building and starting Docker containers...${NC}"
docker compose down > /dev/null 2>&1
docker compose up -d
echo -e "${GREEN}✓ Containers started${NC}"

echo ""
echo -e "${BLUE}[5/5] Waiting for services to be ready...${NC}"
sleep 10

# Check if API is responding
for i in {1..30}; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is responding${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${YELLOW}[!] API took too long to start. Check logs with: docker compose logs app${NC}"
        exit 1
    fi
    echo "    Waiting... (attempt $i/30)"
    sleep 1
done

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Email Classification API Ready!${NC}"
echo "=========================================="
echo ""
echo "API Endpoint: http://localhost:5000"
echo ""
echo "Test the API:"
echo ""
echo "1. Health check:"
echo "   curl http://localhost:5000/health"
echo ""
echo "2. Process an email:"
echo "   curl -X POST http://localhost:5000/api/emails/processar \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"assunto\": \"Erro no sistema\", \"conteudo\": \"Precisamos de suporte urgente para o módulo de pagamentos.\"}'"
echo ""
echo "3. List processed emails:"
echo "   curl http://localhost:5000/api/emails"
echo ""
echo "Documentation:"
echo "  - API Docs:      README-BACKEND.md"
echo "  - Architecture:  ARCHITECTURE.md"
echo "  - Summary:       IMPLEMENTATION-SUMMARY.md"
echo ""
echo "View logs:"
echo "  docker compose logs -f app      # API logs"
echo "  docker compose logs -f db       # Database logs"
echo ""
echo "Stop services:"
echo "  docker compose down"
echo ""
echo "=========================================="
