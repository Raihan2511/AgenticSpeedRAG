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
            # print(f"🌐 Downloading from URL: {source[:30]}...") # Optional logging
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
    # Use 'collection_exists' check if you want to avoid recreating every time,
    # but for development, recreate_collection is fine to reset data.
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

    print("✅ Databases Ready. Starting processing loop...")

    # 5. Get Data Source
    # Currently: Reading files from the local folder
    # Future: You could replace this list with URLs from a CSV file!
    data_items = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(data_items)} items to process...")

    batch_vectors = []
    batch_payloads = []
    batch_ids = []
    
    # 6. Processing Loop
    for idx, item_string in enumerate(data_items):
        try:
            # A. Load the Image (Smart Logic)
            image = load_image(item_string)
            if image is None:
                continue

            # B. Generate Vector (The Real Vibe)
            # We pass the PIL Image object directly to the model
            vector = vision_model.get_embedding(image)
            
            # C. Prepare Metadata (The Facts)
            raw_caption = f"image of {item_string}" 
            clean_caption = meta_cleaner.clean_for_postgres(raw_caption)

            # D. Batching Data (for Qdrant)
            batch_ids.append(idx)
            batch_vectors.append(vector)
            # Store the filename/URL in Qdrant payload so we can retrieve it later
            batch_payloads.append({"filename": item_string, "caption": clean_caption})
            
            # E. Insert into Postgres (Immediate)
            await pg_conn.execute("""
                INSERT INTO image_metadata (id, filename, caption)
                VALUES ($1, $2, $3)
                ON CONFLICT (filename) DO NOTHING
            """, idx, item_string, clean_caption)
            
            # F. Upload Batch to Qdrant (Every 50 items)
            if len(batch_ids) >= 50:
                q_client.upsert(
                    collection_name="speedrag_images",
                    points=Batch(
                        ids=batch_ids,
                        vectors=batch_vectors,
                        payloads=batch_payloads
                    )
                )
                print(f"Uploaded batch ending at ID {idx}")
                batch_ids, batch_vectors, batch_payloads = [], [], []

        except Exception as e:
            print(f"Skipping {item_string}: {e}")

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