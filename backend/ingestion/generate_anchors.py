import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Define the "Concepts" we want the router to understand
# We use multiple phrases to make the anchor robust
INTENTS = {
    "visual_anchor": [
        "show me a picture of", 
        "find an image of", 
        "search for", 
        "look up a photo", 
        "visual search"
    ],
    "chitchat_anchor": [
        "hello", 
        "how are you", 
        "good morning", 
        "tell me a joke", 
        "who are you"
    ]
}

OUTPUT_DIR = "models/anchors"

def generate_anchors():
    print("🧠 Loading SigLIP Text Model...")
    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for intent_name, phrases in INTENTS.items():
        print(f"⚙️ Generating anchor for: {intent_name}...")
        
        # A. Tokenize all phrases
        inputs = tokenizer(phrases, padding="max_length", return_tensors="pt")
        
        # B. Get Vectors for all phrases
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
        
        # C. Normalize them (Just like we did for images!)
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
        # D. THE SECRET SAUCE: Average them together
        # We take the mean of all 5 phrases to find the "center" of the meaning
        avg_vector = outputs.mean(dim=0)
        
        # E. Re-normalize the final average (Physics requirement)
        avg_vector = avg_vector / avg_vector.norm(p=2, dim=-1, keepdim=True)

        # F. Save as .npy
        save_path = os.path.join(OUTPUT_DIR, f"{intent_name}.npy")
        np.save(save_path, avg_vector.numpy())
        print(f"✅ Saved {save_path}")

if __name__ == "__main__":
    generate_anchors()