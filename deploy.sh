#!/bin/bash

# 🚀 Quick Deployment Script for Streamlit Cloud

echo "🔍 Checking prerequisites..."

# Check if git is initialized
if [ ! -d .git ]; then
    echo "⚠️  Git not initialized. Initializing..."
    git init
fi

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "⚠️  requirements.txt not found. Creating from requirements_streamlit.txt..."
    cp requirements_streamlit.txt requirements.txt
fi

# Check if .gitignore exists
if [ ! -f .gitignore ]; then
    echo "⚠️  .gitignore not found. Please create one!"
fi

echo ""
echo "📦 Files ready for deployment:"
echo "  ✅ streamlit_app.py"
echo "  ✅ test_matching_score.py"
echo "  ✅ prompts.py"
echo "  ✅ requirements.txt"
echo "  ✅ .streamlit/config.toml"
echo ""

echo "🔐 IMPORTANT: Before deploying, make sure you have:"
echo "  1. OpenRouter API Key"
echo "  2. Langfuse Public Key"
echo "  3. Langfuse Secret Key"
echo "  4. Langfuse Host URL"
echo ""

echo "📝 Next steps:"
echo "  1. Push this code to GitHub:"
echo "     git add ."
echo "     git commit -m 'Prepare for Streamlit Cloud deployment'"
echo "     git push origin main"
echo ""
echo "  2. Go to: https://share.streamlit.io"
echo "  3. Click 'New app'"
echo "  4. Select your repository"
echo "  5. Set main file: tests/test_matching/streamlit_app.py"
echo "  6. Add secrets in Advanced settings"
echo "  7. Deploy!"
echo ""
echo "✨ Your app will be live at: https://YOUR_APP_NAME.streamlit.app"

