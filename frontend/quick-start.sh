#!/bin/bash

# Frontend Quick Start Script

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Food Allergy AI Agent - Next.js Frontend Setup    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if in frontend directory
if [ ! -f "package.json" ]; then
    echo -e "${YELLOW}Error: package.json not found!${NC}"
    echo "Please run this script from the 'frontend' directory"
    exit 1
fi

# Step 1: Install dependencies
echo -e "${BLUE}Step 1: Installing dependencies...${NC}"
npm install
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 2: Setup environment
echo -e "${BLUE}Step 2: Setting up environment...${NC}"
if [ ! -f ".env.local" ]; then
    cp .env.local.example .env.local
    echo -e "${GREEN}✓ Created .env.local${NC}"
else
    echo -e "${GREEN}✓ .env.local already exists${NC}"
fi
echo ""

# Step 3: Build project
echo -e "${BLUE}Step 3: Building Next.js project...${NC}"
npm run build
echo -e "${GREEN}✓ Build completed${NC}"
echo ""

# Step 4: Success message
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           ✓ Setup Complete!                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}Next steps:${NC}"
echo ""
echo -e "${YELLOW}1. Start the development server:${NC}"
echo "   npm run dev"
echo ""
echo -e "${YELLOW}2. Open your browser:${NC}"
echo "   http://localhost:3000"
echo ""
echo -e "${YELLOW}3. Make sure the backend API is running:${NC}"
echo "   http://localhost:8000"
echo ""
echo -e "${BLUE}Available commands:${NC}"
echo "  npm run dev      - Start development server with hot reload"
echo "  npm run build    - Build for production"
echo "  npm start        - Start production server"
echo "  npm run lint     - Run ESLint"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "  - Frontend README: frontend/README.md"
echo "  - Setup Guide: FRONTEND_SETUP.md"
echo "  - Summary: NEXTJS_FRONTEND_SUMMARY.md"
echo ""
