import os
import pickle
import json
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
import torch
from pathlib import Path

class OfflineSemanticSearch:
    """
    A complete offline semantic search system for medical/healthcare text.
    
    Features:
    - Download and cache models locally
    - Build searchable collections from sentence lists
    - Perform fast similarity search with confidence scores
    - Support for both bi-encoder and hybrid reranking approaches
    """
    
    def __init__(self, models_dir: str = "models", collections_dir: str = "collections"):
        """
        Initialize the semantic search system.
        
        Args:
            models_dir: Directory to store downloaded models
            collections_dir: Directory to store built collections
        """
        self.models_dir = Path(models_dir)
        self.collections_dir = Path(collections_dir)
        
        # Create directories if they don't exist
        self.models_dir.mkdir(exist_ok=True)
        self.collections_dir.mkdir(exist_ok=True)
        
        # Available models configuration
        self.available_models = {
            "pubmedbert": {
                "name": "NeuML/pubmedbert-base-embeddings",
                "type": "bi-encoder",
                "description": "PubMed literature specialist - excellent for medical terminology"
            },
            "clinicalbert": {
                "name": "emilyalsentzer/Bio_ClinicalBERT", 
                "type": "bi-encoder",
                "description": "Clinical notes specialist - good for hospital/clinical text"
            },
            "biobert": {
                "name": "dmis-lab/biobert-v1.1",
                "type": "bi-encoder", 
                "description": "General biomedical specialist - broad medical knowledge"
            },
            "reranker": {
                "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "type": "cross-encoder",
                "description": "High-precision reranker - improves accuracy significantly"
            }
        }
        
        self.loaded_models = {}
        
    def download_and_cache_models(self, model_keys: List[str] = None):
        """
        Download and cache models locally for offline use.
        
        Args:
            model_keys: List of model keys to download. If None, downloads all available models.
        """
        if model_keys is None:
            model_keys = list(self.available_models.keys())
            
        print("📥 Downloading and caching models for offline use...")
        
        for key in model_keys:
            if key not in self.available_models:
                print(f"❌ Model '{key}' not found in available models")
                continue
                
            model_info = self.available_models[key]
            model_path = self.models_dir / key
            
            print(f"\n🔄 Processing {key} ({model_info['name']})...")
            print(f"   Description: {model_info['description']}")
            
            try:
                if model_info['type'] == 'bi-encoder':
                    # Download bi-encoder model
                    model = SentenceTransformer(model_info['name'])
                    model.save(str(model_path))
                    print(f"   ✅ Bi-encoder model cached to: {model_path}")
                    
                elif model_info['type'] == 'cross-encoder':
                    # Download cross-encoder model
                    model = CrossEncoder(model_info['name'])
                    model.save(str(model_path))
                    print(f"   ✅ Cross-encoder model cached to: {model_path}")
                    
            except Exception as e:
                print(f"   ❌ Error downloading {key}: {str(e)}")
                continue
                
        print(f"\n🎉 Model caching complete! Models stored in: {self.models_dir}")
        
    def load_model(self, model_key: str):
        """
        Load a model from local cache.
        
        Args:
            model_key: Key of the model to load
            
        Returns:
            Loaded model object
        """
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
            
        if model_key not in self.available_models:
            raise ValueError(f"Model '{model_key}' not found in available models")
            
        model_path = self.models_dir / model_key
        model_info = self.available_models[model_key]
        
        if not model_path.exists():
            print(f"⚠️  Model '{model_key}' not found locally. Downloading...")
            self.download_and_cache_models([model_key])
            
        try:
            if model_info['type'] == 'bi-encoder':
                model = SentenceTransformer(str(model_path))
            elif model_info['type'] == 'cross-encoder':
                model = CrossEncoder(str(model_path))
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")
                
            self.loaded_models[model_key] = model
            print(f"✅ Loaded {model_key} from cache")
            return model
            
        except Exception as e:
            print(f"❌ Error loading {model_key}: {str(e)}")
            # Fallback to download from HuggingFace
            print("🔄 Trying to download from HuggingFace...")
            if model_info['type'] == 'bi-encoder':
                model = SentenceTransformer(model_info['name'])
            else:
                model = CrossEncoder(model_info['name'])
            self.loaded_models[model_key] = model
            return model
    
    def build_collection(self, sentences: List[str], collection_name: str, 
                        model_key: str = "pubmedbert", force_rebuild: bool = False):
        """
        Build a searchable collection from a list of sentences.
        
        Args:
            sentences: List of sentences to index
            collection_name: Name for the collection
            model_key: Model to use for encoding
            force_rebuild: Whether to force rebuild if collection exists
        """
        collection_path = self.collections_dir / f"{collection_name}.pkl"
        
        if collection_path.exists() and not force_rebuild:
            print(f"📚 Collection '{collection_name}' already exists. Use force_rebuild=True to rebuild.")
            return
            
        print(f"🔨 Building collection '{collection_name}' with {len(sentences)} sentences...")
        print(f"📊 Using model: {model_key} ({self.available_models[model_key]['name']})")
        
        # Load the model
        model = self.load_model(model_key)
        
        # Encode all sentences
        print("🔄 Encoding sentences...")
        embeddings = model.encode(sentences, convert_to_tensor=True, device='cpu', show_progress_bar=True)
        
        # Save collection
        collection_data = {
            'sentences': sentences,
            'embeddings': embeddings.cpu().numpy(),
            'model_key': model_key,
            'model_name': self.available_models[model_key]['name'],
            'collection_size': len(sentences)
        }
        
        with open(collection_path, 'wb') as f:
            pickle.dump(collection_data, f)
            
        print(f"✅ Collection '{collection_name}' built successfully!")
        print(f"📁 Saved to: {collection_path}")
        print(f"📊 Collection size: {len(sentences)} sentences")
        
    def load_collection(self, collection_name: str) -> Dict:
        """
        Load a previously built collection.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            Dictionary containing collection data
        """
        collection_path = self.collections_dir / f"{collection_name}.pkl"
        
        if not collection_path.exists():
            raise FileNotFoundError(f"Collection '{collection_name}' not found at {collection_path}")
            
        with open(collection_path, 'rb') as f:
            collection_data = pickle.load(f)
            
        print(f"📚 Loaded collection '{collection_name}'")
        print(f"📊 Size: {collection_data['collection_size']} sentences")
        print(f"🤖 Model: {collection_data['model_name']}")
        
        return collection_data
    
    def search_bi_encoder(self, query: str, collection_name: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search using bi-encoder approach (fast, good accuracy).
        
        Args:
            query: Search query
            collection_name: Name of the collection to search
            top_k: Number of top results to return
            
        Returns:
            List of (sentence, confidence_score) tuples
        """
        print(f"🔍 Searching with bi-encoder approach...")
        
        # Load collection
        collection = self.load_collection(collection_name)
        
        # Load the model used for this collection
        model = self.load_model(collection['model_key'])
        
        # Encode query
        query_embedding = model.encode(query, convert_to_tensor=True, device='cpu')
        
        # Load collection embeddings
        collection_embeddings = torch.tensor(collection['embeddings'])
        
        # Calculate similarities
        similarities = util.cos_sim(query_embedding, collection_embeddings)[0]
        
        # Get top results
        top_results = torch.topk(similarities, k=min(top_k, len(collection['sentences'])))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            sentence = collection['sentences'][idx.item()]
            confidence = score.item()
            results.append((sentence, confidence))
            
        return results
    
    def search_hybrid_reranking(self, query: str, collection_name: str, 
                               top_k: int = 5, retrieval_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search using hybrid reranking approach (slower, higher accuracy).
        
        Args:
            query: Search query
            collection_name: Name of the collection to search
            top_k: Number of final results to return
            retrieval_k: Number of candidates to retrieve before reranking
            
        Returns:
            List of (sentence, confidence_score) tuples
        """
        print(f"🔍 Searching with hybrid reranking approach...")
        
        # Step 1: Fast retrieval with bi-encoder
        print("   🚀 Step 1: Fast retrieval...")
        initial_results = self.search_bi_encoder(query, collection_name, retrieval_k)
        
        # Step 2: Rerank with cross-encoder
        print("   🎯 Step 2: Precision reranking...")
        reranker = self.load_model('reranker')
        
        # Prepare pairs for reranking
        pairs = [[query, sentence] for sentence, _ in initial_results]
        
        # Get reranking scores
        rerank_scores = reranker.predict(pairs)
        
        # Combine and sort
        reranked_results = []
        for i, (sentence, _) in enumerate(initial_results):
            reranked_results.append((sentence, rerank_scores[i]))
            
        # Sort by reranking score and return top_k
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_results[:top_k]
    
    def search(self, query: str, collection_name: str, method: str = "hybrid", 
               top_k: int = 5, show_details: bool = True) -> List[Tuple[str, float]]:
        """
        Main search function with multiple approaches.
        
        Args:
            query: Search query
            collection_name: Name of the collection to search
            method: Search method ('bi-encoder' or 'hybrid')
            top_k: Number of results to return
            show_details: Whether to print detailed results
            
        Returns:
            List of (sentence, confidence_score) tuples
        """
        print(f"\n{'='*60}")
        print(f"🔍 SEMANTIC SEARCH")
        print(f"{'='*60}")
        print(f"Query: '{query}'")
        print(f"Collection: {collection_name}")
        print(f"Method: {method}")
        print(f"Top K: {top_k}")
        print(f"{'='*60}")
        
        if method == "bi-encoder":
            results = self.search_bi_encoder(query, collection_name, top_k)
        elif method == "hybrid":
            results = self.search_hybrid_reranking(query, collection_name, top_k)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bi-encoder' or 'hybrid'")
            
        if show_details:
            print(f"\n📊 SEARCH RESULTS:")
            print(f"{'='*60}")
            for i, (sentence, score) in enumerate(results, 1):
                print(f"{i}. Score: {score:.4f}")
                print(f"   Text: {sentence}")
                print()
                
        return results
    
    def list_collections(self):
        """List all available collections."""
        collections = list(self.collections_dir.glob("*.pkl"))
        
        print(f"\n📚 AVAILABLE COLLECTIONS:")
        print(f"{'='*60}")
        
        if not collections:
            print("No collections found.")
            return
            
        for collection_path in collections:
            collection_name = collection_path.stem
            try:
                collection = self.load_collection(collection_name)
                print(f"• {collection_name}")
                print(f"  Size: {collection['collection_size']} sentences")
                print(f"  Model: {collection['model_name']}")
                print()
            except Exception as e:
                print(f"• {collection_name} (Error loading: {e})")
                
    def list_models(self):
        """List all available models."""
        print(f"\n🤖 AVAILABLE MODELS:")
        print(f"{'='*60}")
        
        for key, info in self.available_models.items():
            model_path = self.models_dir / key
            cached = "✅ Cached" if model_path.exists() else "❌ Not cached"
            
            print(f"• {key} ({info['type']})")
            print(f"  HuggingFace: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Status: {cached}")
            print()


def main():
    """Example usage of the OfflineSemanticSearch system."""
    
    # Initialize the search system
    search_system = OfflineSemanticSearch()
    
    # Example medical procedure sentences
    medical_procedures = [
        "Chronic Kidney Disease Stage 1 with normal GFR",
        "Chronic Kidney Disease Stage 2 with mild decrease in GFR", 
        "Chronic Kidney Disease Stage 3A with moderate decrease in GFR",
        "Chronic Kidney Disease Stage 4 with severe decrease in GFR",
        "Chronic Kidney Disease Stage 5 requiring dialysis",
        "Type 1 Diabetes Mellitus with diabetic nephropathy",
        "Type 2 Diabetes Mellitus with diabetic nephropathy",
        "Type 2 Diabetes Mellitus with diabetic retinopathy",
        "Heart Failure with reduced ejection fraction",
        "Heart Failure with preserved ejection fraction",
        "Hospital-acquired pneumonia",
        "Community-acquired pneumonia",
        "Essential Hypertension Stage 1",
        "Essential Hypertension Stage 2"
    ]
    
    print("🚀 OFFLINE SEMANTIC SEARCH DEMO")
    print("="*50)
    
    # Step 1: Download and cache models
    print("\n1. Downloading and caching models...")
    search_system.download_and_cache_models(['pubmedbert', 'reranker'])
    
    # Step 2: Build collection
    print("\n2. Building searchable collection...")
    search_system.build_collection(
        sentences=medical_procedures,
        collection_name="medical_procedures",
        model_key="pubmedbert"
    )
    
    # Step 3: Perform searches
    print("\n3. Performing searches...")
    
    # Test queries
    test_queries = [
        "CKD Stage 4 treatment",
        "Type 2 diabetes kidney problems", 
        "Heart failure low ejection fraction"
    ]
    
    for query in test_queries:
        # Bi-encoder search
        results_bi = search_system.search(
            query=query,
            collection_name="medical_procedures", 
            method="bi-encoder",
            top_k=3
        )
        
        # Hybrid search
        results_hybrid = search_system.search(
            query=query,
            collection_name="medical_procedures",
            method="hybrid", 
            top_k=3
        )
        
        print(f"\n📊 COMPARISON FOR: '{query}'")
        print("-" * 50)
        print("Bi-encoder vs Hybrid results:")
        for i in range(min(3, len(results_bi))):
            bi_sentence, bi_score = results_bi[i]
            hybrid_sentence, hybrid_score = results_hybrid[i]
            print(f"{i+1}. Bi-encoder: {bi_score:.4f} | Hybrid: {hybrid_score:.4f}")
            print(f"   Same result: {'✅' if bi_sentence == hybrid_sentence else '❌'}")
    
    # Step 4: Show system info
    print("\n4. System information...")
    search_system.list_models()
    search_system.list_collections()


if __name__ == "__main__":
    main() 