import os
import asyncio
import requests
from io import BytesIO
from PIL import Image
import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Batch

# Import your processors
from backend.ingestion.processors.image_processor import SigLIPProcessor
from backend.ingestion.processors.metadata_processor import MetadataProcessor

# Configuration
QDRANT_URL = "http://localhost:6333"
POSTGRES_DSN = "postgresql://admin:rootpassword@localhost:5432/speedrag_db"
IMAGE_FOLDER = "data/raw_images"

def load_image(source: str):
    """
    Smart Loader: Handles both Web URLs and Local Files.
    Returns: A PIL Image object (ready for the AI).
    """
    try:
        # CASE A: It's a Web URL
        if source.startswith("http://") or source.startswith("https://"):
            # print(f"🌐 Downloading from URL: {source[:30]}...") 
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
            
        # CASE B: It's a Local File (Filename or Path)
        else:
            # If it's just a filename, assume it's in the IMAGE_FOLDER
            if not os.path.isabs(source) and not os.path.exists(source):
                source = os.path.join(IMAGE_FOLDER, source)
                
            if not os.path.exists(source):
                print(f"❌ File not found: {source}")
                return None
            return Image.open(source).convert("RGB")
            
    except Exception as e:
        print(f"⚠️ Error loading image ({source}): {e}")
        return None

async def main():
    print("🚀 Starting Ingestion Pipeline...")
    
    # 1. Initialize Processors
    print("🧠 Loading AI Models...")
    vision_model = SigLIPProcessor() 
    meta_cleaner = MetadataProcessor()
    
    # 2. Connect to Databases
    q_client = QdrantClient(url=QDRANT_URL)
    pg_conn = await asyncpg.connect(POSTGRES_DSN)
    
    # 3. Setup Qdrant Collection
    q_client.recreate_collection(
        collection_name="speedrag_images",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    # 4. Setup Postgres Table
    await pg_conn.execute("""
        CREATE TABLE IF NOT EXISTS image_metadata (
            id SERIAL PRIMARY KEY,
            filename TEXT UNIQUE,
            caption TEXT,
            search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', caption)) STORED
        );
        CREATE INDEX IF NOT EXISTS idx_fulltext ON image_metadata USING GIN(search_vector);
    """)
    # Clear old postgres data to match Qdrant reset
    await pg_conn.execute("TRUNCATE image_metadata RESTART IDENTITY;")

    print("✅ Databases Ready. Preparing Data Sources...")

    # =========================================================
    # 5. CONSOLIDATE DATA SOURCES (Hybrid Logic)
    # =========================================================
    
    # Source A: The "Old Way" (Local Folder Scan)
    # Just gets filenames like "screenshot1.png"
    if os.path.exists(IMAGE_FOLDER):
        local_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        local_files = []
    
    # Source B: The "New Way" (Curated List with Manual Descriptions)
    # ---------------------------------------------------------
    # 💡 EDIT THIS LIST TO ADD YOUR OWN CUSTOM DATA
    # ---------------------------------------------------------
    curated_items = [
        # Example 1: An external URL
        {
            "path": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Python_royal_35.JPG/800px-Python_royal_35.JPG",
            "text": "A royal python snake coiled up on a branch" 
        },
        # Example 2: A local file you want to give a better description to
        # (Make sure this file exists in data/raw_images/)
        {
            "path": "Screenshot 2026-01-21 123653.png", 
            "text": "Outlook error popup 0x800700b appearing on Windows 11 desktop"
        },

        {
            # A reliable URL that allows bots
            "path": "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png",
            "text": "The official Python programming language logo with yellow and blue snakes" 
        }
    ]
    
    # Combine them into one Master List
    master_queue = []
    
    # Add Auto-Detected files
    for f in local_files:
        # Check if this file is already covered by a Manual entry (to avoid duplicates)
        is_covered = any(item['path'] == f for item in curated_items)
        if not is_covered:
            master_queue.append({"type": "auto", "path": f})
        
    # Add Manual files
    for item in curated_items:
        master_queue.append({"type": "manual", "path": item["path"], "text": item["text"]})
        
    print(f"🚀 Processing {len(master_queue)} items ({len(local_files)} Local found, {len(curated_items)} Manual added)...")

    batch_vectors = []
    batch_payloads = []
    batch_ids = []
    
    # 6. Processing Loop
    for idx, entry in enumerate(master_queue):
        try:
            # A. Load Image (Works for both URL and Local because load_image is smart)
            image_source = entry["path"]
            image = load_image(image_source)
            
            if image is None:
                continue

            # B. Generate Vector (The Visual Brain)
            vector = vision_model.get_embedding(image)
            
            # C. Handle Metadata (The Logic Switch)
            if entry["type"] == "manual":
                # CASE 1: Curated Data -> Use the text YOU wrote
                print(f"[{idx}] Processing Manual: {image_source[:30]}...")
                raw_caption = entry["text"]
                
            else:
                # CASE 2: Local File -> Use the filename as a guess
                print(f"[{idx}] Processing Auto: {image_source}...")
                raw_caption = f"image of {image_source}" 

            # D. Clean & Store (Shared Logic)
            clean_caption = meta_cleaner.clean_for_postgres(raw_caption)

            # Batching for Qdrant
            batch_ids.append(idx)
            batch_vectors.append(vector)
            
            # Store the Path (URL or Filename) so the frontend can find it
            batch_payloads.append({"filename": image_source, "caption": clean_caption})
            
            # Insert into Postgres
            await pg_conn.execute("""
                INSERT INTO image_metadata (id, filename, caption)
                VALUES ($1, $2, $3)
                ON CONFLICT (filename) DO NOTHING
            """, idx, image_source, clean_caption)
            
            # E. Upload Batch (Every 50 items)
            if len(batch_ids) >= 50:
                q_client.upsert(
                    collection_name="speedrag_images",
                    points=Batch(ids=batch_ids, vectors=batch_vectors, payloads=batch_payloads)
                )
                print(f"Uploaded batch ending at ID {idx}")
                batch_ids, batch_vectors, batch_payloads = [], [], []

        except Exception as e:
            print(f"Skipping {entry['path']}: {e}")

    # Upload remaining batch (The final cleanup)
    if batch_ids:
        q_client.upsert(
            collection_name="speedrag_images",
            points=Batch(
                ids=batch_ids, 
                vectors=batch_vectors, 
                payloads=batch_payloads
            )
        )

    print("🎉 Ingestion Complete!")

if __name__ == "__main__":
    asyncio.run(main())