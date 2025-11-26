# 🎯 Candidate Matching Streamlit App

A powerful AI-powered candidate evaluation system that matches job postings with candidate CVs using LLM-based analysis.

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud

See **[QUICK_DEPLOY.md](./QUICK_DEPLOY.md)** for fast deployment (5 minutes)

Or see **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** for detailed instructions

---

## ✨ Features

- 📋 **Rubric Extraction** - Automatically extracts evaluation criteria from job postings
- 📊 **Candidate Scoring** - Scores candidates against extracted criteria
- 🎯 **Matching Score** - Calculates weighted matching score (0-100)
- 📝 **Qualification Note** - Generates comprehensive HTML-formatted assessment
- 🔍 **Langfuse Tracking** - Full observability of LLM calls, costs, and performance
- ⚡ **Caching** - Smart caching for rubric extraction
- 🎨 **Model Selection** - Choose between Claude Haiku, Gemini Flash, or Gemini Flash Lite
- 📤 **PDF Upload** - Upload CV PDFs or paste text directly

---

## 📁 Files

| File | Description |
|------|-------------|
| `streamlit_app.py` | Main Streamlit application |
| `test_matching_score.py` | Core matching logic and LLM calls |
| `prompts.py` | Prompt templates |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Streamlit configuration |
| `.gitignore` | Git ignore rules |
| `DEPLOYMENT_GUIDE.md` | Full deployment guide |
| `QUICK_DEPLOY.md` | Quick deployment guide |
| `deploy.sh` | Deployment checklist script |

---

## 🔐 Required Secrets

You'll need these API keys:

- **OpenRouter API Key** - Get from [openrouter.ai/keys](https://openrouter.ai/keys)
- **Langfuse Public Key** - Get from [cloud.langfuse.com](https://cloud.langfuse.com)
- **Langfuse Secret Key** - Get from [cloud.langfuse.com](https://cloud.langfuse.com)
- **Langfuse Host** - Usually `https://cloud.langfuse.com`

### Local Development

Create `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_PUBLIC_KEY = "pk-lf-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_SECRET_KEY = "sk-lf-xxxxxxxxxxxxxxxxxxxxx"
LANGFUSE_HOST = "https://cloud.langfuse.com"
```

⚠️ **Never commit this file!** It's already in `.gitignore`

### Streamlit Cloud

Add secrets in the Streamlit Cloud dashboard under "Settings" → "Secrets"

---

## 🎨 Available Models

- **Claude Haiku 4.5** - Fast, accurate, balanced (default)
- **Gemini Flash 2.5** - Google's fast model
- **Gemini Flash Lite** - Ultra-fast, cost-effective

---

## 📊 Langfuse Integration

The app tracks:
- ✅ All LLM calls (rubric extraction, scoring, qualification)
- ✅ Token usage and costs
- ✅ Response times (pure LLM time, excluding observability overhead)
- ✅ Prompt versions
- ✅ Session grouping

View traces at: [cloud.langfuse.com](https://cloud.langfuse.com)

---

## 🐛 Troubleshooting

### Module Not Found
```bash
pip install -r requirements.txt
```

### Secrets Not Found
Check that secrets are properly configured in `.streamlit/secrets.toml` (local) or Streamlit Cloud dashboard (deployed)

### PDF Extraction Fails
The app tries both PyPDF2 and pdfplumber. If both fail, paste the CV text directly.

---

## 📚 Documentation

- **Quick Deploy**: [QUICK_DEPLOY.md](./QUICK_DEPLOY.md)
- **Full Guide**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
- **Deployment Script**: `./deploy.sh`

---

## 🤝 Contributing

This is a Wiggli Labs project. For questions or issues, contact the development team.

---

## 📄 License

Proprietary - Wiggli Labs

---

## 🎉 Ready to Deploy!

Follow the guides and your app will be live in minutes! 🚀

