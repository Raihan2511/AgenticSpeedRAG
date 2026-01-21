from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
from backend.app.config import settings

# Import all your "Brains"
from backend.app.services.vector_engine import VectorEngine
from backend.app.services.postgres_engine import PostgresEngine
from backend.app.services.semantic_router import SemanticRouter
from backend.app.services.llm_engine import LLMEngine

router = APIRouter()

# Global instances (The Nervous System)
v_engine = None
pg_engine = None
s_router = None
llm_engine = None

class SearchRequest(BaseModel):
    query: str

def reciprocal_rank_fusion(list_a, list_b, k=60):
    """
    Merges two ranked lists using the RRF algorithm.
    """
    scores = {}
    
    # Process List A (Vector Search)
    for rank, item_id in enumerate(list_a):
        if item_id not in scores: scores[item_id] = 0
        scores[item_id] += 1 / (k + rank)
        
    # Process List B (Keyword Search)
    for rank, item_id in enumerate(list_b):
        if item_id not in scores: scores[item_id] = 0
        scores[item_id] += 1 / (k + rank)
    
    # Sort by highest score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:10]] # Return top 10 IDs

@router.on_event("startup")
async def startup_event():
    global v_engine, pg_engine, s_router, llm_engine
    print("🔌 Connecting Neural Pathways...")
    
    # 1. Start Vector Engine (The Eyes)
    v_engine = VectorEngine()
    
    # 2. Start Router (The Gatekeeper)
    s_router = SemanticRouter(vector_engine=v_engine)
    
    # 3. Start Postgres (The Memory)
    pg_engine = PostgresEngine()
    await pg_engine.connect()
    
    # 4. Start LLM Engine (The Voice)
    llm_engine = LLMEngine()
    
    print("✅ System Online and Ready.")

@router.post("/search")
async def hybrid_search(request: SearchRequest):
    # 1. Vectorize the query (Convert text to math)
    query_vec = v_engine.embed(request.query)
    
    # 2. ASK THE ROUTER
    intent = s_router.decide(query_vec)
    
    # 3. If Router says "chitchat", STOP HERE.
    if intent == "chitchat":
        return {
            "intent": "chitchat", 
            "context": [],
            "message": "Hello! I'm ready to help you find visual data. Try asking for a specific diagram or image."
        }

    # 4. EXECUTE HYBRID SEARCH (If it's a visual query)
    # Run both engines in parallel for speed ⚡
    task_vec = v_engine.search_by_vector(query_vec)
    task_kw = pg_engine.search_keywords(request.query)
    
    vec_ids, kw_ids = await asyncio.gather(task_vec, task_kw)
    
    # 5. Rank Fusion (Merge results)
    final_ids = reciprocal_rank_fusion(vec_ids, kw_ids)
    
    if not final_ids:
        return {
            "intent": "visual_search", 
            "context": [], 
            "message": "I looked through the database, but I couldn't find any images matching that description."
        }

    # 6. Fetch Metadata
    raw_results = await pg_engine.fetch_by_ids(final_ids)
    
    # 7. ENRICH RESULTS (Create URLs)
    enriched_results = []
    # This assumes images are hosted at /images endpoint we set up in main.py
    base_url = "http://localhost:8000/images/" 
    
    for item in raw_results:
        # Construct the clickable link
        full_url = base_url + item['filename']
        
        enriched_results.append({
            "id": item['id'],
            "url": full_url,              # <--- The link for the user/frontend
            "description": item['caption'], # <--- The text for the LLM
            "filename": item['filename']
        })
    
    # 8. ✨ GENERATE VOICE RESPONSE (The Agentic Step)
    # We pass the formatted results to the LLM to summarize
    print("💬 Synthesizing answer...")
    ai_message = await llm_engine.synthesize_response(request.query, enriched_results)
    
    return {
        "intent": "visual_search", 
        "context": enriched_results,
        "message": ai_message 
    }