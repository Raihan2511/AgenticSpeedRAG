# from qdrant_client import AsyncQdrantClient
# from transformers import AutoTokenizer
# import onnxruntime as ort
# import numpy as np
# import os

# class VectorEngine:
#     def __init__(self):
#         print("🚀 Initializing Vector Search Engine...")
        
#         # 1. Connect to Qdrant (The Database)
#         self.client = AsyncQdrantClient(url="http://localhost:6333")
        
#         # 2. Load the Tokenizer (To chop up text)
#         # We use the same one from the ingestion phase!
#         self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        
#         # 3. Load the ONNX Model (The Fast Math)
#         # We assume you have exported the model to 'models/onnx/siglip_text.onnx'
#         # (If not, we can use the standard PyTorch version for now)
#         try:
#             self.session = ort.InferenceSession("models/onnx/siglip_text.onnx")
#             self.use_onnx = True
#             print("✅ Loaded ONNX Text Model (Fast Mode)")
#         except:
#             print("⚠️ ONNX model not found. Using standard PyTorch (Slower).")
#             from transformers import AutoModel
#             self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
#             self.use_onnx = False

#     def embed(self, query_text: str):
#         # A. Vectorize the Text
#         if self.use_onnx:
#             # ONNX Logic (Advanced/Fast)
#             inputs = self.tokenizer(query_text, return_tensors="np", padding="max_length", truncation=True, max_length=64)
#             # Run inference
#             outputs = self.session.run(None, dict(inputs))
#             # Normalize
#             vector = outputs[0][0] / np.linalg.norm(outputs[0][0])
#             vector = vector.tolist()
#         else:
#             # Standard PyTorch Logic (Easier fallback)
#             import torch
#             inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)
#             with torch.no_grad():
#                 outputs = self.model.get_text_features(**inputs)
#             outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
#             vector = outputs[0].tolist()
#         return vector

#     async def search_by_vector(self, vector: list, limit: int = 10):
#         # B. Execute Search in Qdrant
#         results = await self.client.search(
#             collection_name="speedrag_images",
#             query_vector=vector,
#             limit=limit
#         )
        
#         # C. Return just the IDs (The "Winners")
#         return [point.id for point in results]

#     async def search(self, query_text: str, limit: int = 10):
#         vector = self.embed(query_text)
#         return await self.search_by_vector(vector, limit)





from qdrant_client import AsyncQdrantClient
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import os

class VectorEngine:
    def __init__(self):
        print("🚀 Initializing Vector Search Engine...")
        
        # 1. Connect to Qdrant
        self.client = AsyncQdrantClient(url="http://localhost:6333")
        
        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        
        # 3. Load ONNX or PyTorch
        try:
            self.session = ort.InferenceSession("models/onnx/siglip_text.onnx")
            self.use_onnx = True
            print("✅ Loaded ONNX Text Model (Fast Mode)")
        except:
            print("⚠️ ONNX model not found. Using standard PyTorch (Slower).")
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
            self.use_onnx = False

    def embed(self, text: str):
        """
        Converts Text -> Vector
        """
        if self.use_onnx:
            # ONNX Logic
            inputs = self.tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=64)
            outputs = self.session.run(None, dict(inputs))
            vector = outputs[0][0] / np.linalg.norm(outputs[0][0])
            return vector.tolist()
        else:
            # PyTorch Logic
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return outputs[0].tolist()

    async def search_by_vector(self, vector: list, limit: int = 10):
        """
        Takes a Vector -> Searches Qdrant using the NEW query_points API
        """
        # 👇 THIS IS THE CRITICAL FIX 👇
        results = await self.client.query_points(
            collection_name="speedrag_images",
            query=vector,  # Renamed from 'query_vector' to 'query'
            limit=limit
        )
        # query_points returns an object with a .points list inside it
        return [point.id for point in results.points]