# 📚 PDF RAG Lab

A modern, beginner-friendly **Retrieval-Augmented Generation (RAG)** application built with Python and Streamlit. Designed specifically for teaching RAG fundamentals to undergraduate Artificial Intelligence and Computer Science students.

> **100% Free & Open Source:** Powered by local TF-IDF vectorization (scikit-learn) and Google's free-tier Gemini API (`gemini-3.5-flash`). Zero paid subscriptions, zero credit card requirements.

---

## 🔬 What is PDF RAG Lab?

Retrieval-Augmented Generation (RAG) is a technique that grounds AI responses in real source documents without fine-tuning or retraining large language models.

This interactive lab demonstrates the entire pipeline across 5 dedicated learning tabs:
1. **📄 Document Explorer:** Upload any text-based PDF, view extracted pages, and inspect clean text.
2. **🗂️ Knowledge Base:** Split text into customizable, overlapping chunks and build an in-memory TF-IDF index.
3. **💬 Ask Questions:** Query your document with natural language and get answers citing specific pages `[Page X]`.
4. **🔎 Retrieval Inspector:** Examine raw similarity scores, ranking metrics, and top retrieved passages for total transparency.
5. **🔬 RAG Pipeline Guide:** Interactive conceptual breakdown comparing RAG vs. Fine-Tuning.

---

## 🏗️ Architecture & Pipeline Flow

```text
[PDF Upload]
     │
     ▼
[Text Extraction]  ──►  pypdf (page-by-page extraction)
     │
     ▼
[Chunking]         ──►  Configurable character window & overlap
     │
     ▼
[Indexing]         ──►  scikit-learn TF-IDF Vectorizer (classical IR)
     │
     ▼
[User Question]    ──►  Cosine similarity retrieval (Top-K passages)
     │
     ├─────────────────────────────────────────┐
     ▼                                         ▼
[Mode 1: Retrieval Only]            [Mode 2: Google Gemini LLM]
 Direct excerpt citations             Grounding prompt + retrieved context
 (Works 100% offline/no key)          Detailed synthesized answer with citations
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/ZainAwanCodes/rag_pdf_lab.git
cd rag_pdf_lab
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure Gemini API Key
To enable LLM answer generation:
1. Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey) (no credit card needed).
2. Create `.streamlit/secrets.toml` in your project folder:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
*(Note: You can also enter or override your key directly in the app's sidebar UI!)*

### 5. Launch the application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌐 Deploy to Streamlit Community Cloud (Free in 2 Minutes)

Deploying this app online for classroom or demo use takes just a few clicks:

1. **Push your code to GitHub** (make sure `.streamlit/secrets.toml` is git-ignored).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **New app** and select your repository:
   - **Repository:** `ZainAwanCodes/rag_pdf_lab`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Expand **Advanced settings** → **Secrets**, and paste:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
5. Click **Deploy!**

For detailed step-by-step instructions with troubleshooting, see **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**.

---

## 📦 Project Structure

```text
rag_pdf_lab/
├── app.py                 # Complete Streamlit application (Frontend + RAG Pipeline)
├── requirements.txt       # Minimal, pinned Python dependencies
├── README.md              # Project overview, architecture, and instructions
├── DEPLOYMENT_GUIDE.md    # Step-by-step cloud deployment guide
├── .gitignore             # Excludes secrets, caches, and virtual environments
└── .streamlit/
    └── secrets.toml       # Local API secrets (Git-ignored: NEVER committed)
```

---

## 🛡️ Security & Privacy

- **Git-Ignored Secrets:** `.streamlit/secrets.toml` is explicitly included in `.gitignore` to prevent secret leaks.
- **Data Privacy:** Your PDF stays in local memory during the session. Only top retrieved chunks are sent to Google Gemini if LLM mode is active.
- **Dual Mode Flexibility:** The application functions fully in **Retrieval Only** mode without requiring any API keys.

---

## 🎓 Classroom Teaching Guide

When presenting this lab to students:
1. **Chunking Tradeoffs:** Adjust the chunk size and overlap sliders to explain the tradeoff between context length and retrieval precision.
2. **Lexical vs. Semantic:** Explain that TF-IDF uses n-gram lexical overlap, serving as a baseline before introducing neural embeddings (e.g. dense vectors).
3. **Hallucination Mitigation:** Show how strict system instructions and context bounding force the LLM to decline questions not covered in the document.

---

## 📄 License

MIT License — Feel free to use and adapt this lab for academic and educational demonstrations.
