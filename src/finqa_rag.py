# src/rag_system.py
import json
from typing import List, Dict, Any
from difflib import SequenceMatcher
import os # For path joining if needed

# You might import DATASET_PATH from config if it becomes complex
# from .config import DATASET_PATH # Assuming config.py is in the same directory or PYTHONPATH is set

class FinancialRAGSystem:
    def __init__(self, dataset_path: str): # Pass dataset_path
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset() # Make load_dataset private convention

    def _load_dataset(self) -> List[Dict]:
        """Load the training dataset from JSON file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content: # Handle empty file
                    print(f"Warning: Dataset file {self.dataset_path} is empty.")
                    return self._get_sample_data()
                
                if content.startswith('['):
                    return json.loads(content)
                else:
                    data = []
                    for line in content.split('\n'):
                        if line.strip():
                            data.append(json.loads(line))
                    return data
        except FileNotFoundError:
            print(f"Warning: Dataset file {self.dataset_path} not found. Using sample data.")
            return self._get_sample_data()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.dataset_path}: {e}. Using sample data.")
            return self._get_sample_data()

    def _get_sample_data(self) -> List[Dict]:
        """Provides sample data if main dataset fails to load."""
        return [{
            "qa": {
                "question": "what was the percentage change in the net cash from operating activities from 2008 to 2009",
                "answer": "14.1%",
                "program": "subtract(206588, 181001), divide(#0, 181001)"
            },
            "pre_text": ["sample pre text"],
            "post_text": ["sample post text"],
            "table": [["sample", "table", "data"]],
            "table_ori": [["original", "table", "format"]],
            "filename": "sample_file.pdf"
        }]

    def calculate_similarity(self, query1: str, query2: str) -> float:
        return SequenceMatcher(None, query1.lower(), query2.lower()).ratio()

    def find_similar_query(self, user_query: str, top_k: int = 1) -> List[Dict]:
        if not self.dataset: # Handle case where dataset is empty even after load attempt
            return []
            
        similarities = []
        for item in self.dataset:
            if 'qa' in item and 'question' in item['qa'] and item['qa']['question']: # Ensure question is not empty
                similarity = self.calculate_similarity(user_query, item['qa']['question'])
                similarities.append({
                    'similarity': similarity,
                    'item': item
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return [s['item'] for s in similarities[:top_k]] # Return items directly

    def extract_context_from_item(self, item: Dict) -> Dict[str, Any]: # Renamed for clarity
        """Extract relevant context from a single dataset item"""
        return {
            'question': item.get('qa', {}).get('question', 'N/A'),
            'answer': item.get('qa', {}).get('answer', 'N/A'),
            'program': item.get('qa', {}).get('program', 'N/A'),
            'pre_text': ' '.join(item.get('pre_text', [])),
            'post_text': ' '.join(item.get('post_text', [])),
            'table_data': item.get('table_ori', item.get('table', [])),
            'filename': item.get('filename', 'Unknown')
        }

# Global instance (or manage instantiation differently, e.g., in main.py)
# For simplicity in tool definition, a global instance can work for this prototype.
# from .config import DATASET_PATH # If DATASET_PATH is needed here
# rag_system_instance = FinancialRAGSystem(dataset_path=DATASET_PATH)
# However, it's cleaner to pass the instance to the tool if the tool is not defined in the same file.
# Let's plan to instantiate it in main.py and pass it to the tool creator.