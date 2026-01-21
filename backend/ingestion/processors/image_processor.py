# # backend\ingestion\processors\image_processor.py
# from PIL import Image
# from transformers import AutoProcessor, AutoModel
# import torch

# class SigLIPProcessor:
#     def __init__(self):
#         print("🧠 Loading SigLIP Vision Model...")
#         # 1. Load the Preprocessor (Resizes/Normalizes images)
#         self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        
#         # 2. Load the Model (The Math Engine)
#         self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        
#         # Optimization: Move to GPU if available, otherwise CPU
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(self.device)

#     def get_embedding(self, image_path: str):
#         """
#         Reads an image and returns a list of 768 floats (The Vector).
#         """
#         try:
#             # A. Open Image
#             image = Image.open(image_path).convert("RGB")
            
#             # B. Preprocess (Convert to Tensor)
#             inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
#             # C. Inference (Run the Model)
#             with torch.no_grad(): # Disable gradient calc for speed
#                 outputs = self.model.get_image_features(**inputs)
            
#             # D. Normalization (Critical for Cosine Similarity!)
#             # We normalize the vector so it has a length of 1.0
#             outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
#             # Convert to standard Python list
#             return outputs[0].cpu().tolist()
            
#         except Exception as e:
#             print(f"❌ Error processing {image_path}: {e}")
#             return None

# # Quick Test
# if __name__ == "__main__":
#     proc = SigLIPProcessor()
#     # Create a dummy image to test
#     dummy = Image.new('RGB', (100, 100), color = 'red')
#     dummy.save("test.jpg")
    
#     vec = proc.get_embedding("test.jpg")
#     print(f"✅ Generated Vector Length: {len(vec)}")
#     # Expected Output: 768





from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch

class SigLIPProcessor:
    def __init__(self):
        print("🧠 Loading SigLIP Vision Model...")
        # 1. Load the Preprocessor (Resizes/Normalizes images)
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        
        # 2. Load the Model (The Math Engine)
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        
        # Optimization: Move to GPU if available, otherwise CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def get_embedding(self, image_input):
        """
        Reads an image (Path or Object) and returns a list of 768 floats (The Vector).
        """
        try:
            # ✅ SMART FIX: Check if input is a String (Path) or Image Object
            if isinstance(image_input, str):
                # It's a file path, so open it
                image = Image.open(image_input).convert("RGB")
            else:
                # It's already a PIL Image object (from our smart pipeline)
                # We just ensure it's RGB
                image = image_input.convert("RGB")
            
            # B. Preprocess (Convert to Tensor)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # C. Inference (Run the Model)
            with torch.no_grad(): # Disable gradient calc for speed
                outputs = self.model.get_image_features(**inputs)
            
            # D. Normalization (Critical for Cosine Similarity!)
            # We normalize the vector so it has a length of 1.0
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # Convert to standard Python list
            return outputs[0].cpu().tolist()
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None

# Quick Test
if __name__ == "__main__":
    proc = SigLIPProcessor()
    # Create a dummy image to test
    dummy = Image.new('RGB', (100, 100), color = 'red')
    
    # Test 1: Passing an Object (What your Pipeline does)
    print("Test 1 (Object):", len(proc.get_embedding(dummy)))
    
    # Test 2: Passing a Path (Old way)
    dummy.save("test.jpg")
    print("Test 2 (Path):", len(proc.get_embedding("test.jpg")))