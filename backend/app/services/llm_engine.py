import requests
import json
import asyncio
from backend.app.config import settings

class LLMEngine:
    def __init__(self):
        print(f"🧠 Initializing LLM Engine (Model: {settings.KRUTRIM_MODEL})...")
        
        # Setup standard HTTP headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.KRUTRIM_API_KEY}"
        }
        # Ensure the URL is correct (Krutrim usually needs /chat/completions appended if base is root)
        # If your .env BASE_URL is "https://cloud.olakrutrim.com/v1", we append "/chat/completions"
        self.api_url = f"{settings.KRUTRIM_BASE_URL}/chat/completions"

    def _make_api_call(self, payload):
        """
        Internal synchronous function that performs the blocking network request.
        """
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json=payload, 
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ API Request Failed: {e}")
            return None

    async def synthesize_response(self, query: str, context: list):
        """
        Async wrapper that offloads the blocking request to a thread.
        """
        # 1. Construct the Context String
        context_str = "\n".join(
            [f"- [Image Found]: {item['url']} (Description: {item['description']})" 
             for item in context]
        )
        
        # 2. Prepare the Payload (Exactly as you requested)
        data = {
            "model": settings.KRUTRIM_MODEL,
            "messages": [
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful visual assistant. Use the provided CONTEXT to answer. "
                        "Always reference specific images found."
                    )
                },
                {
                    "role": "user", 
                    "content": f"User Query: {query}\n\nContext Found:\n{context_str}"
                }
            ],
            "max_tokens": 300,
            "temperature": 0.3
        }

        # 3. Execute in a separate thread to keep the server fast ⚡
        # This converts the blocking 'requests' call into a non-blocking awaitable
        result = await asyncio.to_thread(self._make_api_call, data)
        
        if result and "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return "I found some images, but I couldn't generate a description right now."