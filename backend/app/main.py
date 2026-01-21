from fastapi import FastAPI, HTTPException
from backend.app.routers import search
from backend.app.services.postgres_engine import PostgresEngine
import uvicorn

app = FastAPI(title="Agentic SpeedRAG", version="1.0")

# Initialize the Engine globally so we can check it
pg_engine = PostgresEngine()

app.include_router(search.router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    await pg_engine.connect()

@app.get("/health")
async def health_check():
    """
    Real Health Check: Pings the DB to ensure we can actually serve traffic.
    """
    try:
        # We try to run a simple query. If this fails, the DB is down.
        if not pg_engine.pool:
            raise Exception("Database pool not initialized")
            
        async with pg_engine.pool.acquire() as conn:
            await conn.execute("SELECT 1")
            
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        # This tells the Load Balancer: "STOP SENDING TRAFFIC!"
        raise HTTPException(status_code=503, detail=f"System Unhealthy: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)