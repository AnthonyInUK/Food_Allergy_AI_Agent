#!/bin/bash

# Run both frontend and backend

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Food Allergy AI Agent - Full Stack Launcher       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -d "frontend" ] || [ ! -f "api_server.py" ]; then
    echo -e "${RED}Error: This script must be run from the project root directory${NC}"
    echo "Expected: frontend/ and api_server.py to be in the current directory"
    exit 1
fi

# Start backend
echo -e "${BLUE}Starting backend API server...${NC}"
if [ -f ".venv/bin/activate" ] || [ -f "foodvenv/bin/activate" ]; then
    echo -e "${YELLOW}Activating Python virtual environment...${NC}"
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        source foodvenv/bin/activate
    fi
    
    echo -e "${YELLOW}Starting FastAPI server (uvicorn api_server:app)...${NC}"
    uvicorn api_server:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}Error: No virtual environment found${NC}"
    echo "Please activate your Python virtual environment first"
    exit 1
fi

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to be ready...${NC}"
sleep 3

# Start frontend
echo -e "${BLUE}Starting Next.js frontend...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

echo -e "${YELLOW}Starting Next.js development server...${NC}"
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

cd ..

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           ✓ Both Servers Running!                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}Services:${NC}"
echo -e "  ${GREEN}✓${NC} Backend API:  http://localhost:8000"
echo -e "  ${GREEN}✓${NC} Frontend:    http://localhost:3000"
echo ""

echo -e "${BLUE}Process IDs:${NC}"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""

echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"

# Trap to clean up processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Servers stopped${NC}"
}

trap cleanup EXIT

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
