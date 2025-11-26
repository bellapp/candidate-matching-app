# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy your Candidate Matching Streamlit app to Streamlit Cloud.

---

## 📋 Prerequisites

1. **GitHub Account** - Your code needs to be in a GitHub repository
2. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **API Keys Ready**:
   - OpenRouter API Key
   - Langfuse Public Key
   - Langfuse Secret Key
   - Langfuse Host URL

---

## 🔧 Step 1: Prepare Your Repository

### 1.1 Create `requirements.txt` in the app directory

Rename or copy `requirements_streamlit.txt` to `requirements.txt`:

```bash
cd /home/abdelaazizbellout/Projects/wiggli-labs-test/tests/test_matching
cp requirements_streamlit.txt requirements.txt
```

### 1.2 Update `requirements.txt` with all dependencies

Make sure it includes:

```txt
streamlit>=1.28.0
requests>=2.31.0
PyPDF2>=3.0.0
python-dotenv>=1.0.0
langfuse>=3.0.0
pdfplumber>=0.10.0
```

### 1.3 Create `.streamlit/config.toml` (Optional - for custom theme)

```bash
mkdir -p .streamlit
```

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 10
```

---

## 📦 Step 2: Push to GitHub

### 2.1 Initialize Git (if not already done)

```bash
cd /home/abdelaazizbellout/Projects/wiggli-labs-test
git init
git add .
git commit -m "Add Streamlit candidate matching app"
```

### 2.2 Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click "New Repository"
3. Name it (e.g., `candidate-matching-app`)
4. Don't initialize with README (you already have code)
5. Click "Create repository"

### 2.3 Push Your Code

```bash
git remote add origin https://github.com/YOUR_USERNAME/candidate-matching-app.git
git branch -M main
git push -u origin main
```

---

## ☁️ Step 3: Deploy to Streamlit Cloud

### 3.1 Sign in to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Authorize Streamlit to access your repositories

### 3.2 Create New App

1. Click **"New app"**
2. Select your repository: `YOUR_USERNAME/candidate-matching-app`
3. Set **Branch**: `main`
4. Set **Main file path**: `tests/test_matching/streamlit_app.py`
5. Click **"Advanced settings"** (IMPORTANT!)

### 3.3 Configure Secrets

In the **Advanced settings**, add your secrets in TOML format:

```toml
# OpenRouter API Key
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxx"

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = "pk-lf-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_SECRET_KEY = "sk-lf-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_HOST = "https://cloud.langfuse.com"

# Optional: Default Model
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
```

### 3.4 Deploy!

1. Click **"Deploy!"**
2. Wait 2-5 minutes for deployment
3. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## 🔐 Step 4: Managing Secrets (Alternative Methods)

### Method 1: Via Streamlit Cloud Dashboard (Recommended)

1. Go to your app dashboard
2. Click **"Settings"** → **"Secrets"**
3. Add secrets in TOML format
4. Click **"Save"**
5. App will automatically restart

### Method 2: Via `.streamlit/secrets.toml` (Local Development Only)

⚠️ **NEVER commit this file to Git!**

Create `.streamlit/secrets.toml` locally:

```toml
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_PUBLIC_KEY = "pk-lf-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_SECRET_KEY = "sk-lf-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_HOST = "https://cloud.langfuse.com"
```

Add to `.gitignore`:

```
.streamlit/secrets.toml
```

---

## 🔄 Step 5: Update Your App Code for Secrets

Your `streamlit_app.py` should read secrets like this:

```python
import streamlit as st
import os

# Try to get from Streamlit secrets first, then fall back to environment variables
try:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))
    LANGFUSE_PUBLIC_KEY = st.secrets.get("LANGFUSE_PUBLIC_KEY", os.getenv("LANGFUSE_PUBLIC_KEY"))
    LANGFUSE_SECRET_KEY = st.secrets.get("LANGFUSE_SECRET_KEY", os.getenv("LANGFUSE_SECRET_KEY"))
    LANGFUSE_HOST = st.secrets.get("LANGFUSE_HOST", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
except Exception:
    # Fall back to environment variables
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
```

---

## 🐛 Step 6: Troubleshooting

### Issue: Module Not Found

**Solution**: Check `requirements.txt` includes all dependencies

```bash
# Test locally first
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Issue: Import Errors

**Solution**: Make sure all imports are relative or properly structured

```python
# If test_matching_score.py is in the same directory
from test_matching_score import extract_rubric_with_llm, score_criteria_with_llm
```

### Issue: Secrets Not Found

**Solution**: 
1. Check secrets are added in Streamlit Cloud dashboard
2. Restart the app after adding secrets
3. Check for typos in secret names

### Issue: File Path Errors

**Solution**: Use relative paths, not absolute paths

```python
# ❌ Bad
with open('/home/user/project/file.txt')

# ✅ Good
with open('file.txt')
```

---

## 📊 Step 7: Monitor Your App

### View Logs

1. Go to your app in Streamlit Cloud
2. Click **"Manage app"**
3. View **"Logs"** tab for errors

### Check Analytics

1. Streamlit Cloud provides basic analytics
2. Use Langfuse for detailed LLM call tracking

---

## 🔄 Step 8: Update Your Deployed App

### Automatic Updates

Streamlit Cloud automatically redeploys when you push to GitHub:

```bash
git add .
git commit -m "Update app"
git push
```

### Manual Reboot

1. Go to app dashboard
2. Click **"Reboot app"** if needed

---

## 🎨 Step 9: Custom Domain (Optional)

### Free Subdomain

Your app gets a free subdomain: `your-app-name.streamlit.app`

### Custom Domain (Paid Plans)

1. Upgrade to Streamlit Cloud Pro
2. Add custom domain in settings
3. Update DNS records

---

## 📝 Checklist Before Deployment

- [ ] `requirements.txt` is complete and tested
- [ ] All secrets are configured in Streamlit Cloud
- [ ] No hardcoded API keys in code
- [ ] `.gitignore` includes `.streamlit/secrets.toml`
- [ ] App runs locally without errors
- [ ] All file paths are relative
- [ ] Imports work correctly
- [ ] README.md is updated with app description

---

## 🚀 Quick Deploy Commands

```bash
# 1. Navigate to project
cd /home/abdelaazizbellout/Projects/wiggli-labs-test/tests/test_matching

# 2. Copy requirements
cp requirements_streamlit.txt requirements.txt

# 3. Test locally
streamlit run streamlit_app.py

# 4. Commit and push
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main

# 5. Deploy on Streamlit Cloud (via web interface)
# Go to: https://share.streamlit.io
```

---

## 🔗 Useful Links

- **Streamlit Cloud**: https://share.streamlit.io
- **Streamlit Docs**: https://docs.streamlit.io
- **Deployment Docs**: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
- **Secrets Management**: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- **Langfuse Docs**: https://langfuse.com/docs

---

## 💡 Pro Tips

1. **Use Caching**: Streamlit Cloud has limited resources, use `@st.cache_data` for expensive operations
2. **Optimize Uploads**: Set `maxUploadSize` in config.toml
3. **Monitor Costs**: Track OpenRouter API usage in Langfuse
4. **Version Control**: Use Git tags for releases
5. **Environment Variables**: Use secrets for all sensitive data

---

## 🎉 You're Done!

Your Streamlit app should now be live and accessible to anyone with the URL!

Share it with your team: `https://your-app-name.streamlit.app`

