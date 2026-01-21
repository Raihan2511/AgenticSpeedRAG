import numpy as np
import os

class SemanticRouter:
    def __init__(self, anchors_path="models/anchors"):
        print("⚡ Initializing Zero-Latency Router...")
        self.anchors = {}
        
        # 1. Load Pre-computed Anchor Vectors
        # These are the "centroids" of your intents (Visual Search vs Chitchat)
        # We assume these .npy files exist (we will create them next!)
        try:
            self.anchors["visual"] = np.load(os.path.join(anchors_path, "visual_anchor.npy"))
            self.anchors["chitchat"] = np.load(os.path.join(anchors_path, "chitchat_anchor.npy"))
        except FileNotFoundError:
            print("⚠️ Anchors not found! Router will default to 'chitchat'.")

    def decide(self, query_vector: list) -> str:
        """
        Input: User's query vector (from SigLIP Text Model)
        Output: 'visual' or 'chitchat'
        """
        if not self.anchors:
            return "chitchat"

        scores = {}
        
        # 2. Convert query to numpy for fast math
        query_np = np.array(query_vector)

        # 3. Calculate Similarity (Dot Product)
        # Since vectors are normalized, Dot Product == Cosine Similarity
        for intent, anchor_vec in self.anchors.items():
            scores[intent] = np.dot(query_np, anchor_vec)

        # 4. The "Winner Takes All" Decision
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]

        # 5. Threshold Gate
        # If the confidence is too low, assume it's just random chat
        if confidence < 0.45:
            return "chitchat"
            
        return best_intent

# Test
if __name__ == "__main__":
    # Mocking a vector for testing
    router = SemanticRouter()
    dummy_vec = [0.1] * 768 
    print(f"Decision: {router.decide(dummy_vec)}")