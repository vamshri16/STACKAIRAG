# Phase 1: Foundation & Environment Setup

**Goal:** Establish the project skeleton so the application can be installed, started, and verified before any RAG logic is written. This phase is purely about engineering hygiene — a clean, reproducible base that any reviewer can clone and run in under a minute.

---

## Step 1 — Create the Python Virtual Environment

**What we did:**
```bash
python3 -m venv .venv
```

**Why:**
A virtual environment isolates this project's Python dependencies from the system Python and from other projects on the machine. Without it, installing packages globally can cause version conflicts, break other projects, or make the project non-reproducible on another machine.

**What it produced:**
```
.venv/
├── bin/           # Python interpreter, pip, uvicorn (after install)
├── include/
├── lib/           # All installed packages go here
└── pyvenv.cfg     # Points to the base Python 3.13 interpreter
```

The `.venv/` directory is listed in `.gitignore` — it is never committed to version control. Every developer recreates it locally by running `python3 -m venv .venv` followed by `pip install -r requirements.txt`.

---

## Step 2 — Define Dependencies (`requirements.txt`)

**What we did:**
Created `requirements.txt` with 11 pinned dependencies:

```
fastapi==0.115.6          # Web framework — async, modern, auto-generates OpenAPI docs
uvicorn[standard]==0.34.0 # ASGI server to run FastAPI (includes uvloop for speed)
pydantic==2.10.4          # Data validation — used in request/response models
pydantic-settings==2.7.1  # Reads environment variables into typed Settings class
python-dotenv==1.0.1      # Loads .env file into os.environ at startup
httpx==0.28.1             # Async HTTP client — will call Mistral AI API
numpy==2.2.1              # Numeric operations — cosine similarity, embeddings storage
PyPDF2==3.0.1             # PDF text extraction (primary)
pdfplumber==0.11.4        # PDF text extraction (fallback for complex layouts)
pytesseract==0.3.13       # OCR wrapper — for scanned/image-based PDFs
python-multipart==0.0.20  # Required by FastAPI for file upload (multipart/form-data)
```

**Why every version is pinned:**
Pinning (e.g. `==0.115.6` instead of `>=0.115`) ensures that every developer, CI runner, and deployment environment installs the exact same versions. Without pinning, a minor update to any library could introduce a breaking change that works on one machine but fails on another.

**What is deliberately NOT in this file:**

| Library | Why it is excluded |
|---|---|
| `faiss-cpu` / `faiss-gpu` | Forbidden — we implement vector search from scratch with NumPy |
| `rank-bm25` | Forbidden — we implement keyword scoring from scratch |
| `sentence-transformers` | Forbidden — no cross-encoder re-ranking |
| `scikit-learn` | Forbidden — no TF-IDF, no sklearn cosine similarity |
| `spacy` | Not needed yet — will be added in a later phase if PII detection requires it |

**How to install:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 3 — Environment Variable Configuration (`.env.example`)

**What we did:**
Created `.env.example` as a template:

```
MISTRAL_API_KEY=your_mistral_api_key_here
DATA_DIR=./data
INDEX_DIR=./indexes
```

**Why two files (`.env.example` vs `.env`):**

| File | Committed to git? | Contains real secrets? | Purpose |
|---|---|---|---|
| `.env.example` | Yes | No — placeholder values only | Shows developers which variables are needed |
| `.env` | No (in `.gitignore`) | Yes — real API keys | Actual runtime configuration, never shared |

A developer clones the repo, copies `.env.example` to `.env`, fills in their real Mistral API key, and the app reads it automatically at startup.

**What each variable controls:**
- `MISTRAL_API_KEY` — Authentication token for the Mistral AI API (embeddings + chat generation)
- `DATA_DIR` — Where uploaded PDF files are stored on disk
- `INDEX_DIR` — Where the NumPy embeddings matrix and chunk metadata are persisted

---

## Step 4 — Application Configuration (`app/config.py`)

**What we did:**
Created a `Settings` class using Pydantic's `BaseSettings`:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mistral_api_key: str = ""
    mistral_embed_model: str = "mistral-embed"
    mistral_chat_model: str = "mistral-large-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieval: int = 20
    top_k_final: int = 5
    similarity_threshold: float = 0.7
    semantic_weight: float = 0.7

    data_dir: str = "./data"
    index_dir: str = "./indexes"

    model_config = {"env_file": ".env"}


settings = Settings()
```

**How it works:**
1. When the module is imported, `Settings()` is instantiated
2. Pydantic reads the `.env` file (if it exists) and maps each variable to the corresponding field
3. Environment variables override `.env` file values (standard 12-factor app behavior)
4. Every field has a sensible default so the app can start even without a `.env` file

**What each setting does:**

| Setting | Default | Purpose |
|---|---|---|
| `mistral_api_key` | `""` (empty) | API key for Mistral AI. Empty string means "not configured" |
| `mistral_embed_model` | `"mistral-embed"` | Which Mistral model to use for generating embeddings |
| `mistral_chat_model` | `"mistral-large-latest"` | Which Mistral model to use for answer generation |
| `mistral_base_url` | `"https://api.mistral.ai/v1"` | Base URL for all Mistral API calls |
| `chunk_size` | `800` | Number of tokens per text chunk when splitting PDFs |
| `chunk_overlap` | `150` | Number of overlapping tokens between consecutive chunks |
| `top_k_retrieval` | `20` | How many candidate chunks to retrieve in the initial search |
| `top_k_final` | `5` | How many chunks to return after re-ranking |
| `similarity_threshold` | `0.7` | Minimum cosine similarity score to include a chunk in results |
| `semantic_weight` | `0.7` | Weight for semantic score in hybrid search (keyword weight = 1 - 0.7 = 0.3) |
| `data_dir` | `"./data"` | Directory for storing uploaded PDF files |
| `index_dir` | `"./indexes"` | Directory for persisting the embeddings matrix and metadata |

**Why `pydantic-settings` instead of raw `os.getenv()`:**
- Type validation — `chunk_size` is guaranteed to be an `int`, not a string
- Single source of truth — all config in one class, not scattered across files
- IDE autocomplete — `settings.chunk_size` is typed, so editors can help
- Documentation — the class itself documents what configuration exists

---

## Step 5 — Package Structure (Empty `__init__.py` Files)

**What we did:**
Created the following directory tree with empty `__init__.py` files:

```
app/
├── __init__.py             # Makes 'app' a Python package
├── api/
│   └── __init__.py         # Makes 'app.api' a Python package
├── services/
│   └── __init__.py         # Makes 'app.services' a Python package
├── models/
│   └── __init__.py         # Makes 'app.models' a Python package
└── utils/
    └── __init__.py         # Makes 'app.utils' a Python package
```

**Why:**
Python requires `__init__.py` files to treat directories as importable packages. Without them, `from app.api import health` would fail with `ModuleNotFoundError`.

**What each package will contain (in later phases):**

| Package | Future contents |
|---|---|
| `app/api/` | FastAPI route handlers (ingestion, query, health) |
| `app/services/` | Business logic (PDF processing, embeddings, vector store, search, LLM client) |
| `app/models/` | Pydantic schemas for request/response validation |
| `app/utils/` | Shared utilities (text processing, PII detection) |

---

## Step 6 — FastAPI Application Entry Point (`app/main.py`)

**What we did:**
Created the FastAPI application instance:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health

app = FastAPI(
    title="RAG Pipeline",
    description="PDF Knowledge Base with Retrieval-Augmented Generation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
```

**Line-by-line explanation:**

1. **`FastAPI(...)`** — Creates the application object. The `title`, `description`, and `version` are used to generate the automatic Swagger/OpenAPI documentation page at `/docs`.

2. **`CORSMiddleware`** — Cross-Origin Resource Sharing. The React frontend (running on `localhost:5173`) needs to make API calls to the backend (running on `localhost:8000`). Without CORS middleware, the browser blocks these cross-origin requests. `allow_origins=["*"]` permits requests from any origin during development.

3. **`app.include_router(health.router)`** — Registers the health check route. FastAPI uses a "router" pattern — each file defines its own `APIRouter` with endpoints, and the main app includes them. This keeps route definitions modular and organized.

**How to run it:**
```bash
uvicorn app.main:app --reload --port 8000
```
- `app.main:app` tells uvicorn: "import `app` from `app/main.py`"
- `--reload` watches for file changes and auto-restarts (development only)
- `--port 8000` binds to port 8000

---

## Step 7 — Health Check Endpoint (`app/api/health.py`)

**What we did:**
Created the first API endpoint:

```python
from fastapi import APIRouter

from app.config import settings

router = APIRouter()


@router.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "mistral_configured": bool(settings.mistral_api_key),
        "data_dir": settings.data_dir,
        "index_dir": settings.index_dir,
    }
```

**What it returns:**
```json
{
    "status": "healthy",
    "mistral_configured": false,
    "data_dir": "./data",
    "index_dir": "./indexes"
}
```

**Why a health check endpoint exists:**
- Confirms the server starts and responds to HTTP requests
- Shows whether the Mistral API key is configured (`mistral_configured: true/false`)
- In production, load balancers and monitoring tools poll `/api/health` to know if the service is alive
- It is the simplest possible endpoint — if this fails, nothing else will work

**Why `mistral_configured` is important:**
The Mistral API key is required for embeddings and answer generation. If it is not set, the health check immediately tells the developer "you forgot to create a `.env` file with your API key" without having to wait until they try to run a query and get a cryptic API error.

---

## Step 8 — Git Ignore Rules (`.gitignore`)

**What we did:**
Created `.gitignore` to prevent sensitive and generated files from being committed:

```
# Virtual environment
.venv/
venv/

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Environment
.env

# Data and indexes (runtime artifacts)
data/
indexes/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

**What each rule prevents:**

| Pattern | What it blocks | Why |
|---|---|---|
| `.venv/`, `venv/` | Virtual environment directory | Large (100+ MB), machine-specific, reproducible from requirements.txt |
| `__pycache__/`, `*.py[cod]` | Python bytecode cache | Generated automatically, differs per Python version |
| `.env` | Environment file with secrets | Contains real API keys — must never be in git |
| `data/` | Uploaded PDF files | User data, potentially large, not part of the codebase |
| `indexes/` | Persisted embeddings and metadata | Runtime artifacts, regenerated from data |
| `.vscode/`, `.idea/` | IDE configuration | Personal preference, differs per developer |
| `.DS_Store` | macOS filesystem metadata | OS artifact, irrelevant to the project |

---

## Step 9 — Initialize Git Repository & First Commit

**What we did:**
```bash
git init
git add .env.example .gitignore IMPLEMENTATION_PLAN.md README.md app/ requirements.txt
git commit -m "Phase 1: Foundation & environment setup"
```

**What was committed (13 files):**
```
.env.example
.gitignore
IMPLEMENTATION_PLAN.md
README.md
app/__init__.py
app/api/__init__.py
app/api/health.py
app/config.py
app/main.py
app/models/__init__.py
app/services/__init__.py
app/utils/__init__.py
requirements.txt
```

**What was NOT committed (correctly excluded by `.gitignore`):**
- `.venv/` — virtual environment (hundreds of files)
- `.DS_Store` — macOS metadata
- No `.env` file existed yet, but it would be excluded if it did

---

## Step 10 — Verification

**What we tested:**

1. **Server starts without errors:**
   ```bash
   .venv/bin/uvicorn app.main:app --port 8000
   ```
   Uvicorn printed `INFO: Application startup complete` — no import errors, no missing dependencies.

2. **Health endpoint returns valid JSON:**
   ```bash
   curl http://localhost:8000/api/health
   ```
   Response:
   ```json
   {
       "status": "healthy",
       "mistral_configured": false,
       "data_dir": "./data",
       "index_dir": "./indexes"
   }
   ```

3. **Swagger documentation page loads:**
   ```bash
   curl -o /dev/null -w "%{http_code}" http://localhost:8000/docs
   ```
   Returned HTTP `200`. The auto-generated API docs are available at `http://localhost:8000/docs`.

4. **Git working tree is clean:**
   ```bash
   git status
   ```
   Output: `nothing to commit, working tree clean` — every file is either committed or correctly ignored.

---

## Files Created in Phase 1

| File | Lines | Purpose |
|---|---|---|
| `requirements.txt` | 11 | Pinned Python dependencies |
| `.env.example` | 3 | Environment variable template |
| `.gitignore` | 27 | Git exclusion rules |
| `app/__init__.py` | 0 | Package marker |
| `app/config.py` | 23 | Centralized typed configuration |
| `app/main.py` | 20 | FastAPI application with CORS |
| `app/api/__init__.py` | 0 | Package marker |
| `app/api/health.py` | 15 | Health check endpoint |
| `app/models/__init__.py` | 0 | Package marker (empty, used in later phases) |
| `app/services/__init__.py` | 0 | Package marker (empty, used in later phases) |
| `app/utils/__init__.py` | 0 | Package marker (empty, used in later phases) |

---

## What Phase 1 Does NOT Include

This phase intentionally excludes all RAG logic:

- No PDF processing
- No text chunking
- No embeddings
- No vector store
- No search (semantic or keyword)
- No LLM integration
- No frontend

These are all addressed in Phases 2 through 12 as defined in `IMPLEMENTATION_PLAN.md`.

---

## How to Reproduce Phase 1 From Scratch

```bash
# Clone the repository
git clone <repo-url>
cd rag-pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your API key
cp .env.example .env
# Edit .env and replace 'your_mistral_api_key_here' with your real key

# Start the server
uvicorn app.main:app --reload --port 8000

# Test
curl http://localhost:8000/api/health
```
