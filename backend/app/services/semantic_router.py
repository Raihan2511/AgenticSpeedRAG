# backend\app\services\semantic_router.py
# import numpy as np
# import os

# class SemanticRouter:
#     def __init__(self, anchors_path="models/anchors"):
#         print("⚡ Initializing Zero-Latency Router...")
#         self.anchors = {}
        
#         # 1. Load Pre-computed Anchor Vectors
#         # These are the "centroids" of your intents (Visual Search vs Chitchat)
#         # We assume these .npy files exist (we will create them next!)
#         try:
#             self.anchors["visual"] = np.load(os.path.join(anchors_path, "visual_anchor.npy"))
#             self.anchors["chitchat"] = np.load(os.path.join(anchors_path, "chitchat_anchor.npy"))
#         except FileNotFoundError:
#             print("⚠️ Anchors not found! Router will default to 'chitchat'.")

#     def decide(self, query_vector: list) -> str:
#         """
#         Input: User's query vector (from SigLIP Text Model)
#         Output: 'visual' or 'chitchat'
#         """
#         if not self.anchors:
#             return "chitchat"

#         scores = {}
        
#         # 2. Convert query to numpy for fast math
#         query_np = np.array(query_vector)

#         # 3. Calculate Similarity (Dot Product)
#         # Since vectors are normalized, Dot Product == Cosine Similarity
#         for intent, anchor_vec in self.anchors.items():
#             scores[intent] = np.dot(query_np, anchor_vec)

#         # 4. The "Winner Takes All" Decision
#         best_intent = max(scores, key=scores.get)
#         confidence = scores[best_intent]

#         # 5. Threshold Gate
#         # If the confidence is too low, assume it's just random chat
#         if confidence < 0.45:
#             return "chitchat"
            
#         return best_intent

# # Test
# if __name__ == "__main__":
#     # Mocking a vector for testing
#     router = SemanticRouter()
#     dummy_vec = [0.1] * 768 
#     print(f"Decision: {router.decide(dummy_vec)}")






import numpy as np

class SemanticRouter:
    def __init__(self, vector_engine):
        print("🚦 Initializing Semantic Router...")
        self.vector_engine = vector_engine
        
        # 1. Define "Chitchat" Phrases (The Stop List)
        # These are the phrases that should NOT trigger a database search.
        self.chitchat_phrases = [
            "hello", "hi", "hey", "good morning", "good evening", 
            "how are you", "sup", "what's up", "nice to meet you",
            "thanks", "thank you", "bye", "goodbye", "cool"
        ]
        
        # 2. Pre-calculate their vectors (The "Reference Book")
        # We do this once at startup so it's super fast later.
        self.routes = []
        for phrase in self.chitchat_phrases:
            vec = self.vector_engine.embed(phrase)
            self.routes.append(vec)
            
    def decide(self, query_vector: list):
        """
        Decides if the query is 'chitchat' or a real 'visual_search'.
        Returns: "chitchat" | "visual_search"
        """
        # 3. Calculate Similarity (Dot Product)
        # We compare the User's Query Vector against all our Chitchat Vectors
        if not self.routes:
            return "visual_search" # Safety fallback
            
        # Convert lists to numpy arrays for fast math
        query_np = np.array(query_vector)
        routes_np = np.array(self.routes)
        
        # Calculate scores (How similar is the query to our phrases?)
        scores = np.dot(routes_np, query_np)
        
        # 4. The Decision Threshold
        # If the best match is > 0.45, we assume it's chitchat.
        # (SigLIP embeddings are normalized, so 0.45 is a solid similarity score)
        best_score = np.max(scores)
        
        if best_score > 0.45:
            print(f"🛑 Router blocked search (Score: {best_score:.2f})")
            return "chitchat"
        else:
            print(f"✅ Router allowed search (Score: {best_score:.2f})")
            return "visual_search"
