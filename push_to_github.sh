#!/bin/bash

# 🚀 Push to GitHub Repository
# Repository: https://github.com/bellapp/candidate-matching-app.git

echo "🔧 Setting up Git repository..."

# Initialize git if not already done
if [ ! -d .git ]; then
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# Add all files
echo "📦 Adding files..."
git add .

# Create initial commit
echo "💾 Creating commit..."
git commit -m "Initial commit: Candidate matching Streamlit app

Features:
- Rubric extraction from job postings
- Candidate scoring with LLM
- Qualification note generation
- Langfuse observability
- Model selection (Claude, Gemini)
- PDF upload support
- Caching for performance"

# Add remote repository
echo "🔗 Adding remote repository..."
git remote add origin https://github.com/bellapp/candidate-matching-app.git 2>/dev/null || git remote set-url origin https://github.com/bellapp/candidate-matching-app.git

# Set main branch
git branch -M main

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Done! Your code is now on GitHub:"
echo "   https://github.com/bellapp/candidate-matching-app"
echo ""
echo "🎯 Next step: Deploy to Streamlit Cloud"
echo "   1. Go to: https://share.streamlit.io"
echo "   2. Click 'New app'"
echo "   3. Select repository: bellapp/candidate-matching-app"
echo "   4. Set main file: streamlit_app.py"
echo "   5. Add secrets in Advanced settings"
echo "   6. Deploy!"

