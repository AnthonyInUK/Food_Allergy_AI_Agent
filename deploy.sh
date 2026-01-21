#!/bin/bash

# Food Allergy AI Agent - Deploy to GitHub & HuggingFace
# Run this script to deploy latest changes

echo "🚀 Starting deployment..."
echo ""

# Add all changes
echo "📦 Adding files..."
git add .

# Check what will be committed
echo ""
echo "📝 Files to commit:"
git status --short

echo ""
read -p "Continue with commit? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Commit with message
    echo ""
    echo "💾 Committing changes..."
    git commit -m "Performance optimization: Multi-tier routing + 6-layer cache + parallel execution

- Fast-path SQL routing (1000x speedup for simple queries)
- Semantic normalization cache (cross-language deduplication)
- Parallel quality graders (30% latency reduction)
- Multi-hop query optimization (skip unnecessary routing)
- Full English UI interface"
    
    # Push to GitHub
    echo ""
    echo "📤 Pushing to GitHub..."
    git push origin main
    
    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "📍 GitHub: Check your repository"
    echo "📍 HuggingFace: https://huggingface.co/spaces/AnthonyInBC/Anthony_space"
    echo ""
    echo "⏳ HuggingFace will auto-sync in ~1-2 minutes"
else
    echo ""
    echo "❌ Deployment cancelled"
fi

