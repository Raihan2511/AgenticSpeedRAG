# Agentic SpeedRAG: Low-Latency Multimodal Retrieval System

**Research Objective:** To achieve sub-300ms retrieval latency on a dataset of 1 million image-text pairs while maintaining high semantic accuracy and agentic reasoning capabilities.

---

## 📖 Table of Contents

1. [Problem Statement](https://www.google.com/search?q=%23-problem-statement)
2. [System Architecture](https://www.google.com/search?q=%23-system-architecture)
3. [Core Methodologies](https://www.google.com/search?q=%23-core-methodologies)
4. [Tech Stack](https://www.google.com/search?q=%23-tech-stack)
5. [Project Structure](https://www.google.com/search?q=%23-project-structure)
6. [Getting Started](https://www.google.com/search?q=%23-getting-started)

---

## 🎯 Problem Statement

Modern multimodal RAG systems often suffer from the "Agentic Bottleneck," where Large Language Model (LLM) reasoning loops introduce 1-3 seconds of latency. This project implements a **"Retrieve-First, Reason-Later"** topology designed to handle **1,000,000 records** with a strict **<300ms** response budget.

### Performance Targets

| **Metric**          | **Constraint** | **Method**                    |
| ------------------------- | -------------------- | ----------------------------------- |
| **Total Latency**   | **≤ 300ms**   | Parallel execution & ONNX Runtime   |
| **Router Decision** | **< 5ms**      | Deterministic Vector Classification |
| **Search Scale**    | **1M Records** | HNSW Indexing + INT8 Quantization   |
| **Recall Quality**  | **High**       | Hybrid Search (Vector + Keyword)    |

---

## 🏛 System Architecture

The system is divided into two distinct asynchronous pipelines: **Offline Ingestion** (The Preparation) and **Online Runtime** (The Reflex Arc).

### 1. The Runtime Logic (The 300ms Loop)

* **Deterministic Routing:** Instead of asking an LLM "should I search?", we map the user's intent mathematically using a lightweight `MiniLM` model.
* **Hybrid Search Engine:** We run two search strategies in parallel:
  * **Semantic Path:** `SigLIP` (ONNX) **$\to$** `Qdrant` (Finds images that *look* like the query).
  * **Precision Path:** `Keyword Extraction` **$\to$** `PostgreSQL` (Finds exact text matches).
* **Late-Binding Synthesis:** The LLM (Gemini/DeepSeek) is only invoked at the very end to synthesize the natural language response, ensuring it never blocks the retrieval process.

### 2. The Bridge (Integration)

* **Diagram:** [Refer to `docs/architecture_hybrid.png`]
* The system utilizes **Reciprocal Rank Fusion (RRF)** to merge the "Vibe-based" results from Qdrant with the "Fact-based" results from PostgreSQL.

---

## 🧠 Core Methodologies

### 1. Semantic Routing (The Zero-Latency Agent)

We replace probabilistic LLM reasoning with deterministic vector similarity. By pre-computing "Intent Anchors" (e.g., specific vectors for "Visual Search" vs. "Chitchat"), we can classify user intent in microseconds.

### 2. Cross-Modal Projection (SigLIP)

We utilize Google's  **SigLIP (Sigmoid Loss for Language Image Pre-training)** . This model projects text and images into a shared semantic vector space, allowing us to query image data using natural language without complex translation layers.

### 3. Decoupled Memory Architecture

To maximize speed, we split memory into two tiers:

* **Fast Memory (RAM):** Qdrant stores *only* compressed vectors (INT8) and IDs.
* Slow Memory (Disk): PostgreSQL stores heavy metadata (Captions, URLs, Logs).
  This prevents "memory bloat" and keeps the search index extremely lightweight.

---

## 🛠 Tech Stack

### Phase 1: Ingestion (Offline)

* **Language:** Python 3.10+
* **Vision Model:** `google/siglip-base-patch16-224`
* **Processing:** `Pillow` (Image), `Transformers` (Embedding)

### Phase 2: Application (Online)

* **API Framework:** `FastAPI` + `Uvicorn` (Async)
* **Inference Engine:** `ONNX Runtime` (Hardware Accelerated)
* **Vector Database:** `Qdrant` (Rust-based, HNSW Index)
* **Metadata Store:** `PostgreSQL` (GIN Index for Full-Text Search)
* **LLM Integration:** `Gemini 1.5 Flash` / `DeepSeek`

---

## 📂 Project Structure

**Plaintext**

```
/backend
├── /ingestion              # PHASE 1: OFFLINE PREP
│   ├── pipeline.py         # ETL Master Script
│   └── processors/         # SigLIP & Cleaning Logic
│
├── /app                    # PHASE 2: ONLINE RUNTIME
│   ├── main.py             # FastAPI Entry Point
│   ├── services/           # Business Logic
│   │   ├── semantic_router.py  # <5ms Decision Engine
│   │   ├── vector_engine.py    # Qdrant Wrapper
│   │   └── postgres_engine.py  # Hybrid Search Wrapper
│   └── routers/            # API Endpoints
│
├── /core                   # Shared Configs (.env, DB connections)
└── /models                 # Stores Quantized ONNX models & Anchors
```

---

## 🚀 Getting Started

### Prerequisites

* Docker & Docker Compose
* Python 3.10+
* GPU (Recommended for Ingestion, Optional for Runtime)

### Quick Start

1. **Infrastructure Up:**
   **Bash**

   ```
   docker-compose up -d
   ```
2. **Ingest Data (Prepare the 1M Index):**
   **Bash**

   ```
   python backend/ingestion/pipeline.py
   ```
3. **Launch API (Start the 300ms Engine):**
   **Bash**

   ```
   uvicorn backend.app.main:app --reload
   ```

---

### 📩 Contact & Research

**Author:** Raihan Uddin

**Focus:** Agentic AI, High-Performance Retrieval, Linux Systems

**Institution:** RKMVERI, Dept. of Computer Science
