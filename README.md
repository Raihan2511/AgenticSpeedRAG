# ⚡ Agentic SpeedRAG: Theoretical Architecture & Implementation
<!-- The Nervous System of a High-Performance AI Agent -->

**Agentic SpeedRAG** represents a paradigm shift in retrieval systems, moving away from monolithic LLM-based reasoning towards a **"Retrieve-First, Reason-Later"** topology. This architecture is designed to eliminate the "Agentic Bottleneck"—the 1-3 second latency penalty incurred when Agents pause to "think" before acting.

Instead, this system implements a **biological reflex arc** for information retrieval, where the decision to search, the execution of the search, and the synthesis of results happen in parallel, deterministic streams before the high-level LLM is ever involved.

---

## 🧠 System Execution Flow (The "Life of a Data Point")

The codebase is organized chronologically: **Offline Learning** (Ingestion) $\to$ **Online Reflex** (Search). Below is the exact order of operations, the files involved, and the **specific tools/libraries** powering each step.

### Phase 1: The Offline Ingestion Layer (Learning)

Before the system can answer questions, it must "learn" the data. This happens asynchronously.

#### 1. The Orchestrator: [`backend/ingestion/pipeline.py`](backend/ingestion/pipeline.py)
*   **The Job:** This is the master control script. It loads images from disk/web and coordinates the sensation-to-memory process.
*   **🛠 Tools & Libraries:**
    *   **`asyncpg`**: Used for high-speed, asynchronous connections to PostgreSQL.
    *   **`qdrant_client`**: Used to push vector batches to the Qdrant database.
    *   **`requests`**: Used to download images if the source is a URL.
    *   **`asyncio`**: Manages the concurrent execution of IO tasks.
*   **The Theory:** **Dual-Write Architecture**. It ensures atomicity; data is either written to *both* Qdrant (Vector) and Postgres (Metadata) or *neither*.

#### 2. The Visual Cortex: [`backend/ingestion/processors/image_processor.py`](backend/ingestion/processors/image_processor.py)
*   **The Job:** Converts raw pixels (RGB) into a mathematical concept (Vector).
*   **🛠 Tools & Libraries:**
    *   **`transformers` (HuggingFace)**: Loads the SigLIP model architecture.
    *   **`torch` (PyTorch)**: Performs the tensor math and matrix multiplications.
    *   **`PIL` (Pillow)**: Reads the raw image bytes and handles resizing/cropping.
*   **The Theory:** **High-Dimensional Projection**. It projects a $224 \times 224$ image into a $768$-dimensional hypersphere using **SigLIP** (Sigmoid Loss for Language Image Pre-training).

#### 3. The Auditory Cortex: [`backend/ingestion/processors/metadata_processor.py`](backend/ingestion/processors/metadata_processor.py)
*   **The Job:** Cleans raw text for the SQL database to ensure perfect keyword matching.
*   **🛠 Tools & Libraries:**
    *   **`re` (Regex)**: The Python standard library for string pattern matching.
*   **The Theory:** **Noise Reduction**. By removing special characters and standardizing whitespace, we maximize the "recall" rate of the exact keyword search later.

---

### Phase 2: The Online Reflex Layer (Thinking)

This is the runtime loop that executes when a user sends a query.

#### 4. The Gateway: [`backend/app/main.py`](backend/app/main.py)
*   **The Job:** The entry point. Initializes the neural engines and exposes the API.
*   **🛠 Tools & Libraries:**
    *   **`fastapi`**: The web framework that handles the HTTP request/response cycle.
    *   **`uvicorn`**: The ASGI server that runs the application.
*   **The How:** It defines a single startup lifecycle hook to connect to the databases *once*, preventing connection overhead on every request.

#### 5. The Thalamus: [`backend/app/services/semantic_router.py`](backend/app/services/semantic_router.py)
*   **The Job:** The very first check. Decides *if* we need to search.
*   **🛠 Tools & Libraries:**
    *   **`numpy`**: Used for ultra-fast, CPU-based vector math (Dot Product). We avoid PyTorch here to save overhead.
*   **The Theory:** **Vector Similarity Thresholding**.
    $$ Score = \mathbf{q} \cdot \mathbf{A}_{visual} $$
    If $Score > 0.45$, we classify as "Visual Search". If lower, we classify as "Chitchat".

#### 6. The Right Brain (Intuition): [`backend/app/services/vector_engine.py`](backend/app/services/vector_engine.py)
*   **The Job:** Finds images that have the same "vibe" or semantic meaning.
*   **🛠 Tools & Libraries:**
    *   **`qdrant_client` (Async)**: Performs the HNSW search query.
    *   **`onnxruntime`**: Runs the SigLIP text encoder. We use ONNX instead of PyTorch here because it is ~2x faster for CPU inference.
*   **The Theory:** **HNSW (Hierarchical Navigable Small World)**. A graph-based algorithm that finds approximate nearest neighbors in logarithmic time ($O(\log N)$).

#### 7. The Left Brain (Logic): [`backend/app/services/postgres_engine.py`](backend/app/services/postgres_engine.py)
*   **The Job:** Finds images that strictly match specific keywords.
*   **🛠 Tools & Libraries:**
    *   **`asyncpg`**: Executes raw SQL queries against the database.
*   **The Theory:** **GIN (Generalized Inverted Index)**.
    *   *Stemming:* The database reduces "Running" to "run".
    *   *Index:* It maps the word "run" to a list of IDs $\{12, 55, 90\}$, allowing for $O(1)$ lookup speed.

#### 8. The Frontal Lobe (Synthesis): [`backend/app/routers/search.py`](backend/app/routers/search.py)
*   **The Job:** Merges the "Vibe" results and the "Fact" results into one final answer.
*   **🛠 Tools & Libraries:**
    *   **`asyncio`**: Runs the Vector Search and Keyword Search *at the same time* (Parallelism).
*   **The Theory:** **Reciprocal Rank Fusion (RRF)**.
    $$ Score(d) = \sum \frac{1}{k + rank(d)} $$
    This formula prioritizes consensus, ensuring the final result is best of both worlds.

---

## 📂 Physical Repository Map

Every file in the codebase corresponds to a specific biological function described above.

```graphql
/backend
├── /ingestion                  # PHASE 1: OFFLINE LEARNING
│   ├── pipeline.py             #    [1] The Orchestrator
│   └── processors/             #
│       ├── image_processor.py  #    [2] Visual Cortex (SigLIP)
│       └── metadata_processor.py #  [3] Auditory Cortex (Cleaning)
│
├── /app                        # PHASE 2: ONLINE REFLEX
│   ├── main.py                 #    [4] Gateway (FastAPI)
│   ├── routers/                #
│   │   └── search.py           #    [8] Frontal Lobe (RRF Fusion)
│   └── services/               #
│       ├── semantic_router.py  #    [5] Thalamus (Decision Gate)
│       ├── vector_engine.py    #    [6] Right Brain (HNSW Search)
│       └── postgres_engine.py  #    [7] Left Brain (GIN Search)
```
