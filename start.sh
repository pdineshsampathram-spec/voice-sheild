#!/bin/bash
# VoiceShield — Quick Start Script
# Starts both backend and frontend in parallel

set -e

echo "🛡️  VoiceShield — Starting..."
echo ""

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Check Python venv
if [ ! -d "$PROJECT_DIR/backend/venv" ]; then
    echo -e "${CYAN}Creating Python virtual environment...${NC}"
    cd "$PROJECT_DIR/backend"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo -e "${GREEN}✓${NC} Python venv found"
fi

# Check Node modules
if [ ! -d "$PROJECT_DIR/frontend/node_modules" ]; then
    echo -e "${CYAN}Installing frontend dependencies...${NC}"
    cd "$PROJECT_DIR/frontend"
    npm install
else
    echo -e "${GREEN}✓${NC} Node modules found"
fi

# Create .env if missing
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo -e "${GREEN}✓${NC} Created .env from template"
fi

echo ""
echo "Starting services..."
echo "  Backend:  http://localhost:8000  (API docs: http://localhost:8000/docs)"
echo "  Frontend: http://localhost:3000"
echo ""

# Start backend
cd "$PROJECT_DIR/backend"
source venv/bin/activate
PYTHONPATH="$PROJECT_DIR/backend" uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# Trap Ctrl+C to kill both
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo 'VoiceShield stopped.'; exit" INT TERM

echo ""
echo -e "${GREEN}🛡️  VoiceShield is running!${NC}"
echo "   Press Ctrl+C to stop."
echo ""

wait
