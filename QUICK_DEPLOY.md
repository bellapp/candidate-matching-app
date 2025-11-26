# 🚀 Quick Deployment to Streamlit Cloud

## ⚡ Fast Track (5 Minutes)

### 1️⃣ Push to GitHub

```bash
cd /home/abdelaazizbellout/Projects/wiggli-labs-test
git add .
git commit -m "Add Streamlit candidate matching app"
git push origin main
```

### 2️⃣ Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Fill in:
   - **Repository**: `YOUR_USERNAME/wiggli-labs-test`
   - **Branch**: `main`
   - **Main file**: `tests/test_matching/streamlit_app.py`

4. Click **"Advanced settings"** and add secrets:

```toml
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_PUBLIC_KEY = "pk-lf-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_SECRET_KEY = "sk-lf-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_HOST = "https://cloud.langfuse.com"
```

5. Click **"Deploy!"**

### 3️⃣ Done! 🎉

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 📋 Files Created for Deployment

✅ `requirements.txt` - Python dependencies
✅ `.streamlit/config.toml` - App configuration
✅ `.gitignore` - Protect secrets
✅ `DEPLOYMENT_GUIDE.md` - Full deployment guide
✅ `deploy.sh` - Quick deployment script

---

## 🔐 Required Secrets

You'll need these API keys:

| Secret Name | Where to Get It |
|------------|----------------|
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `LANGFUSE_PUBLIC_KEY` | [cloud.langfuse.com](https://cloud.langfuse.com) → Settings → API Keys |
| `LANGFUSE_SECRET_KEY` | [cloud.langfuse.com](https://cloud.langfuse.com) → Settings → API Keys |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` (or your self-hosted URL) |

---

## 🐛 Common Issues

### "Module not found"
→ Check `requirements.txt` includes all dependencies

### "Secrets not found"
→ Add secrets in Streamlit Cloud dashboard under "Settings" → "Secrets"

### "Import error"
→ Make sure `test_matching_score.py` and `prompts.py` are in the same directory

---

## 📚 Full Documentation

For detailed instructions, see: **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)**

---

## 💡 Pro Tip

Test locally before deploying:

```bash
cd tests/test_matching
streamlit run streamlit_app.py
```

If it works locally, it will work on Streamlit Cloud! ✨

