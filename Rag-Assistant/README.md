# 📚 FYP Handbook RAG Assistant

**Assignment:** Text-only RAG pipeline for FAST-NUCES FYP Handbook  
**Course:** Final Year Project (FYP)  
**Institution:** FAST-NUCES

---

## 🎯 Overview

A Retrieval-Augmented Generation (RAG) system that answers student questions about the FAST-NUCES Final Year Project process using the BS FYP Handbook 2023 as the sole knowledge base.

### Key Features

✅ **Text-only RAG pipeline** (no LLM API required)  
✅ **Sentence-BERT embeddings** (all-MiniLM-L6-v2)  
✅ **FAISS vector store** for efficient similarity search  
✅ **Page citations** in every answer  
✅ **Similarity threshold** (0.25) to prevent hallucinations  
✅ **CLI and Streamlit interfaces**

---

## 🏗️ System Architecture

```
PDF Input → Text Extraction (per page) → Chunking (350 words, 30% overlap)
    ↓
Embedding (all-MiniLM-L6-v2) → FAISS Index (Cosine Similarity)
    ↓
Query → Retrieve Top-5 → Threshold Check → Answer with Citations
```

### Assignment Requirements Met

- ✅ Chunk size: 250-400 words (using 350)
- ✅ Overlap: 20-40% (using 30%)
- ✅ Metadata: page, section_hint, chunk_id
- ✅ Embeddings: all-MiniLM-L6-v2
- ✅ Top-k: 5 chunks by cosine similarity
- ✅ Threshold: 0.25
- ✅ Grounding: Page references in answers

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd fyp-handbook-rag
```

### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Handbook PDF

Place your FYP Handbook PDF in the `data/` folder:

```bash
data/handbook.pdf
```

---

## 🚀 Usage

### Step 1: Run Ingestion Pipeline

Process the PDF, create chunks, and build FAISS index:

```bash
python src/ingest.py
```

**What it does:**

- Extracts text page by page
- Creates overlapping chunks (350 words, 30% overlap)
- Generates embeddings with all-MiniLM-L6-v2
- Builds FAISS index with cosine similarity
- Saves to disk

**Output files:**

- `data/chunks.json` - Human-readable chunks
- `data/embeddings/faiss.index` - FAISS vector index
- `data/embeddings/meta.pkl` - Chunk metadata
- `data/embeddings/model_name.txt` - Model identifier

---

### Step 2: Ask Questions

#### Option A: CLI (Command Line)

**Single question:**

```bash
python src/ask.py "What are the formatting requirements?"
```

**Interactive mode:**

```bash
python src/ask.py
```

Then type questions interactively. Type `help` for sample questions.

---

#### Option B: Streamlit UI (Recommended)

```bash
streamlit run src/app.py
```

Open browser at `http://localhost:8501`

**Features:**

- Input box for questions
- "Ask" button
- Answer panel with page citations
- Collapsible "Sources" section showing retrieved chunks
- Sample questions in sidebar

---

## 📝 Validation Questions

Test your system with these 6 questions from the assignment:

1. **"What headings, fonts, and sizes are required in the FYP report?"**

   - Expected: Times New Roman 11 for body, Arial for headings, specific sizes

2. **"What margins and spacing do we use?"**

   - Expected: Top 1.5", Bottom 1.0", Left 2.0", Right 1.0", line spacing 1.5

3. **"What are the required chapters of a Development FYP report?"**

   - Expected: Intro, research, vision, SRS, iterations, implementation, manual, refs, appendices

4. **"What are the required chapters of an R&D FYP report?"**

   - Expected: Intro, literature review, approach, implementation, validation, results, conclusions

5. **"How should endnotes like 'Ibid.' and 'op. cit.' be used?"**

   - Expected: Ibid. for immediate repetition, op. cit. after interruptions

6. **"What goes into the Executive Summary and Abstract?"**
   - Expected: Abstract 50-125 words, Executive summary 1-2 pages

---

## ⚙️ Configuration

Edit `src/config.py` to adjust settings:

| Parameter              | Value            | Description                 |
| ---------------------- | ---------------- | --------------------------- |
| `CHUNK_SIZE_WORDS`     | 350              | Target chunk size (250-400) |
| `OVERLAP_PERCENTAGE`   | 0.30             | Chunk overlap (20-40%)      |
| `TOP_K`                | 5                | Chunks to retrieve          |
| `SIMILARITY_THRESHOLD` | 0.25             | Min cosine similarity       |
| `EMBEDDING_MODEL`      | all-MiniLM-L6-v2 | Sentence-BERT model         |

---

## 📊 Project Structure

```
fyp-handbook-rag/
│
├── data/
│   ├── handbook.pdf              # Input PDF (place here)
│   ├── chunks.json               # Generated chunks (human-readable)
│   └── embeddings/
│       ├── faiss.index           # FAISS vector index
│       ├── meta.pkl              # Chunk metadata
│       └── model_name.txt        # Model identifier
│
├── src/
│   ├── config.py                 # Configuration & hyperparameters
│   ├── utils.py                  # Helper functions
│   ├── ingest.py                 # Ingestion pipeline
│   ├── ask.py                    # CLI interface (main)
│   ├── app.py                    # Streamlit UI (optional)
│   └── prompt_template.txt       # Prompt template
│
├── logs/
│   └── prompt_log.txt            # Logged queries & answers
│
├── docs/
│   ├── design_note.pdf           # Technical documentation
│   └── screenshots/              # UI screenshots
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔍 How It Works

### 1. **Text Extraction**

- Uses `pdfplumber` to extract text page by page
- Preserves page numbers for citations
- Cleans text (removes extra spaces, normalizes newlines)

### 2. **Chunking**

- Splits text into 350-word chunks
- 30% overlap between consecutive chunks
- Extracts section hints (headings)
- Stores metadata: `page`, `section`, `chunk_id`, `word_count`

### 3. **Embedding**

- Uses `all-MiniLM-L6-v2` from Sentence-Transformers
- 384-dimensional dense vectors
- Normalized for cosine similarity

### 4. **Indexing**

- FAISS `IndexFlatIP` (Inner Product after L2 normalization = Cosine)
- Exact search (no approximation)
- Fast retrieval even for large handbooks

### 5. **Retrieval**

- Encode user query with same model
- Search FAISS for top-5 most similar chunks
- Filter by threshold (0.25)
- Return "I don't have that in the handbook" if below threshold

### 6. **Answer Generation**

- No LLM needed (as per assignment)
- Format retrieved chunks with page citations
- Preserve original handbook text (grounding)
- Log all queries to `prompt_log.txt`

---

## 📋 Deliverables Checklist

### 1. Code Files ✅

- [x] `src/ingest.py` - Parse & index
- [x] `src/ask.py` - Retrieve & answer (CLI)
- [x] `src/app.py` - Streamlit UI (optional)
- [x] `src/utils.py` - Helper functions
- [x] `src/config.py` - Configuration
- [x] `src/prompt_template.txt` - Prompt template

### 2. Design Note (1-2 pages PDF) 📄

Create a document including:

- Chunking settings (350 words, 30% overlap)
- Model used (all-MiniLM-L6-v2)
- Top-k value (5)
- Similarity threshold (0.25)
- Screenshots of 2 example Q&A with page citations

### 3. Prompt Log 📝

- Automatically saved to `logs/prompt_log.txt`
- Contains all questions, retrieved context, and answers
- Includes page references and similarity scores

---

## 🛡️ Grounding & Safety

### Preventing Hallucinations

✅ **Similarity threshold:** Rejects queries with score < 0.25  
✅ **Scope limitation:** Only answers from handbook content  
✅ **Page citations:** Every answer includes source pages  
✅ **No generation:** Returns actual handbook text (no paraphrasing)

### Example Output

```
Question: What margins are required?

Answer:
The required margins are:
- Top: 1.5 inches
- Bottom: 1.0 inches
- Left: 2.0 inches
- Right: 1.0 inches

Line spacing should be 1.5 and paragraph spacing 6 pt. (p. 12)

📄 Referenced Pages: 12
🎯 Confidence: HIGH
```

---

## 🐛 Troubleshooting

### Issue: "FAISS index not found"

**Solution:** Run ingestion first: `python src/ingest.py`

### Issue: "No text extracted from PDF"

**Solution:** Ensure PDF is text-based (not scanned images). Place at `data/handbook.pdf`

### Issue: Poor retrieval quality

**Solution:**

- Check if chunks are meaningful in `data/chunks.json`
- Adjust `CHUNK_SIZE_WORDS` or `OVERLAP_PERCENTAGE` in `config.py`
- Re-run ingestion after changes

### Issue: "I don't have that in the handbook" for valid questions

**Solution:**

- Lower `SIMILARITY_THRESHOLD` in `config.py` (e.g., 0.20)
- Increase `TOP_K` for more chunks
- Check if question wording matches handbook terminology

---

## 📊 System Performance

Typical metrics (on handbook ~50 pages):

- **Ingestion time:** 1-2 minutes
- **Index size:** ~500KB
- **Query time:** <1 second
- **Memory usage:** ~500MB (model + index)

---

## 🔧 Advanced Configuration

### Using Different Embedding Models

Edit `config.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better quality
# OR
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Faster
```

Then re-run: `python src/ingest.py`

### Adjusting Chunk Strategy

```python
CHUNK_SIZE_WORDS = 300  # Smaller chunks
OVERLAP_PERCENTAGE = 0.40  # More context
```

---

## 📚 References

- **Sentence-BERT:** [https://www.sbert.net/](https://www.sbert.net/)
- **FAISS:** [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- **Assignment:** FAST-NUCES FYP Handbook RAG Project

---

## 👥 Contributors

- **Your Name** - FAST-NUCES Student

---

## 📄 License

This project is for educational purposes as part of FAST-NUCES FYP coursework.

---

## 📧 Contact

For questions or issues:

- Email: your.email@example.com
- GitHub Issues: [link-to-repo-issues]

---

**Built with:** Sentence-BERT • FAISS • Streamlit  
**For:** FAST-NUCES Final Year Project  
**Year:** 2023-2024
