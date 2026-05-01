#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Food Allergy AI Agent Frontend...${NC}"

# Check if running from the correct directory
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found. Please run this script from the frontend directory."
    exit 1
fi

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
npm install

# Create .env.local if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo -e "${BLUE}📝 Creating .env.local...${NC}"
    cp .env.local.example .env.local
    echo -e "${GREEN}✓ Created .env.local${NC}"
fi

echo -e "${BLUE}🔨 Building project...${NC}"
npm run build

echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo -e "${BLUE}To start the development server:${NC}"
echo "  npm run dev"
echo ""
echo -e "${BLUE}To start the production server:${NC}"
echo "  npm start"
echo ""
echo -e "${BLUE}API Server should be running at:${NC}"
echo "  http://localhost:8000"
echo ""
echo -e "${BLUE}Frontend will be available at:${NC}"
echo "  http://localhost:3000"
