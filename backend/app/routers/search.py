from fastapi import APIRouter
from pydantic import BaseModel
import asyncio

# Import our services
from backend.app.services.semantic_router import SemanticRouter
from backend.app.services.vector_engine import VectorEngine
from backend.app.services.postgres_engine import PostgresEngine

router = APIRouter()

# Initialize Services
# (In a real app, use Dependency Injection, but this is fine for now)
s_router = SemanticRouter()
v_engine = VectorEngine()
pg_engine = PostgresEngine()

class SearchRequest(BaseModel):
    query: str

@router.on_event("startup")
async def startup():
    await pg_engine.connect()

@router.post("/search")
async def hybrid_search(request: SearchRequest):
    # 1. Vectorize the Query ONCE (<30ms)
    # We add a helper to VectorEngine to just give us the vector
    # (We need to update VectorEngine to expose this, let's assume it has .embed(text))
    query_vec = v_engine.embed(request.query)
    
    # 2. Router Decision (<5ms)
    intent = s_router.decide(query_vec)
    print(f"🧠 Intent Detected: {intent}")
    
    if intent == "chitchat":
        return {"intent": "chitchat", "context": []}

    # 3. Parallel Search Execution (<50ms)
    # Fire both engines at the same time!
    task_vec = v_engine.search_by_vector(query_vec)
    task_kw = pg_engine.search_keywords(request.query)
    
    # Wait for both to finish
    vec_ids, kw_ids = await asyncio.gather(task_vec, task_kw)
    
    # 4. Rank Fusion (The Hybrid Magic)
    final_ids = reciprocal_rank_fusion(vec_ids, kw_ids)
    
    # 5. Fetch Content (<10ms)
    results = await pg_engine.fetch_by_ids(final_ids)
    
    return {"intent": "visual_search", "context": results}

def reciprocal_rank_fusion(list_a, list_b, k=60):
    """
    Combines two lists of IDs based on their Rank (position), not score.
    Score = 1 / (Rank + k)
    """
    scores = {}
    
    # Process List A (Vector Results)
    for rank, item_id in enumerate(list_a):
        scores[item_id] = scores.get(item_id, 0) + (1 / (rank + k))
        
    # Process List B (Keyword Results)
    for rank, item_id in enumerate(list_b):
        scores[item_id] = scores.get(item_id, 0) + (1 / (rank + k))
        
    # Sort by combined score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_ids[:5] # Return top 5 winners