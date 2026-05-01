#!/bin/bash

# Update HuggingFace Space with latest local code

echo "🔄 Updating HuggingFace Space..."
echo ""

cd /Users/anthony/Desktop/llm/foodAIAgent

# Check if huggingface remote exists
if ! git remote | grep -q "huggingface"; then
    echo "➕ Adding HuggingFace remote..."
    git remote add huggingface https://huggingface.co/spaces/AnthonyInBC/Food_Allergic_Agent
fi

# Fetch remote changes first
echo "📥 Fetching remote changes..."
git fetch huggingface

# Check if we need to merge
echo ""
echo "🔀 Merging remote changes (if any)..."

# Use --allow-unrelated-histories if needed
git pull huggingface main --allow-unrelated-histories --no-edit || {
    echo ""
    echo "⚠️  Merge conflicts detected. Resolving by keeping local version..."
    # Keep local version for all files
    git checkout --ours .
    git add .
    git commit -m "Merge: Keep local latest version"
}

# Push to HuggingFace
echo ""
echo "📤 Pushing to HuggingFace..."
git push huggingface main

echo ""
echo "✅ Update complete!"
echo "📍 Your Space: https://huggingface.co/spaces/AnthonyInBC/Food_Allergic_Agent"
echo "⏳ Build will start automatically in ~1 minute"


