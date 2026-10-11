# 🚀 Deployment Guide — PDF RAG Lab on Streamlit Community Cloud

This guide provides a comprehensive walkthrough for deploying the **PDF RAG Lab** to Streamlit Community Cloud for free.

---

## Prerequisites

- A [GitHub](https://github.com) account (free)
- A [Streamlit Community Cloud](https://share.streamlit.io) account (free, sign in with GitHub)
- A [Google AI Studio](https://aistudio.google.com/app/apikey) account (free tier, no credit card required)

---

## Step 1 — Get Your Free Gemini API Key

1. Navigate to [Google AI Studio API Keys](https://aistudio.google.com/app/apikey).
2. Click **Create API Key**.
3. Copy your API key.
4. Keep your key confidential — never commit it to public repositories.

---

## Step 2 — Push Code to GitHub

Your repository is already structured for instant deployment with the primary entry point `app.py` located at the root.

```bash
git init
git add app.py requirements.txt README.md DEPLOYMENT_GUIDE.md .gitignore
git commit -m "feat: complete PDF RAG Lab application"
git branch -M main
git remote add origin https://github.com/ZainAwanCodes/rag_pdf_lab.git
git push -u origin main
```

> ⚠️ **Verification:** Confirm that `.streamlit/secrets.toml` is **not** staged or committed. It is safely protected by `.gitignore`.

---

## Step 3 — Deploy on Streamlit Community Cloud

1. Visit **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **New app** (or **Create app**).
3. Select your repository details:
   - **Repository:** `ZainAwanCodes/rag_pdf_lab`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings** (bottom of the configuration modal).
5. In the **Secrets** text box, paste your Gemini API key:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
6. Click **Save** and then click **Deploy!**

Deployment will typically complete within 1–2 minutes.

---

## Step 4 — Verify Live Deployment

1. Once the deployment finishes, Streamlit will give you a public URL (e.g., `https://rag-pdf-lab-xxxx.streamlit.app`).
2. Open the URL.
3. Check the sidebar under **⚙️ Settings → 🤖 Answer Mode**:
   - You should see: `✅ GEMINI_API_KEY detected & active`.
4. Upload a sample lecture or research paper PDF.
5. Head to the **🗂️ Knowledge Base** tab and click **🔨 Build Knowledge Base**.
6. Switch to **💬 Ask Questions** and test natural language queries.

---

## Troubleshooting & FAQ

| Issue | Solution |
| :--- | :--- |
| **Secrets not detected on Cloud** | Go to your app dashboard → App Settings → **Secrets**, ensure key name is exact: `GEMINI_API_KEY = "..."`. |
| **Model Rate Limit (429)** | Gemini free tier allows up to 15 requests per minute. Wait a moment or switch to **Retrieval Only** mode. |
| **Scanned PDF error** | The PDF parser extracts text. Scanned document images require OCR software before upload. |
| **Package installation issues** | Ensure `requirements.txt` is in the root directory alongside `app.py`. |

---

## Updating Your Cloud App

Any time you make improvements or changes, simply push them to GitHub:
```bash
git add app.py
git commit -m "update: prompt and chunking enhancements"
git push origin main
```
Streamlit Community Cloud will automatically detect the commit and redeploy in seconds!
