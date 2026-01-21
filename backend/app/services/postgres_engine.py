import asyncpg
import os

class PostgresEngine:
    def __init__(self):
        print("🐘 Initializing Postgres Hybrid Engine...")
        self.dsn = "postgresql://admin:rootpassword@localhost:5432/speedrag_db"
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.dsn)

    async def fetch_by_ids(self, ids: list):
        """
        Job 1: The "Handoff"
        Takes IDs from Qdrant -> Returns Captions from Postgres
        """
        if not ids:
            return []
            
        async with self.pool.acquire() as conn:
            # We use ANY($1) to fetch multiple IDs in a single query (Fast!)
            rows = await conn.fetch(
                "SELECT id, filename, caption FROM image_metadata WHERE id = ANY($1)",
                ids
            )
            return [dict(row) for row in rows]

    async def search_keywords(self, query_text: str, limit: int = 5):
        """
        Job 2: The "Keyword Search" (Hybrid Part)
        Finds exact matches like 'XJ-900' or 'Error 404'
        """
        async with self.pool.acquire() as conn:
            # plainto_tsquery parses "red car" into "red & car" for us
            rows = await conn.fetch(
                """
                SELECT id, ts_rank(search_vector, query) as score
                FROM image_metadata, plainto_tsquery('english', $1) query
                WHERE search_vector @@ query
                ORDER BY score DESC LIMIT $2
                """,
                query_text, limit
            )
            return [row['id'] for row in rows]