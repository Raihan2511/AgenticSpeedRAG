# metadata_processor.py
import re

class MetadataProcessor:
    def clean_for_postgres(self, text: str) -> str:
        """
        Prepares text for PostgreSQL Full-Text Search (TSVECTOR).
        Removes special characters that might break the SQL query.
        """
        # 1. Lowercase for consistency
        text = text.lower()
        
        # 2. Remove non-alphanumeric characters (keep numbers for XJ-900!)
        # We keep spaces to separate words.
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # 3. Remove extra whitespace
        text = " ".join(text.split())
        
        return text

# Quick Test
if __name__ == "__main__":
    processor = MetadataProcessor()
    raw = "ERROR: Connection Refused!! [Code: 404]"
    print(processor.clean_for_postgres(raw))
    # Output: error connection refused code 404