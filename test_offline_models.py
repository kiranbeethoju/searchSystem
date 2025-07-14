#!/usr/bin/env python3
"""
Test script to verify offline model functionality
"""

from offline_semantic_search import OfflineSemanticSearch
import os

def test_offline_models():
    """Test if models are properly downloaded and loaded from offline folders"""
    
    print("🧪 TESTING OFFLINE MODEL FUNCTIONALITY")
    print("=" * 50)
    
    # Initialize system
    search_system = OfflineSemanticSearch()
    
    # Test 1: Download and cache models
    print("\n1. Testing model download and caching...")
    search_system.download_and_cache_models(['pubmedbert', 'reranker'])
    
    # Check if models were saved
    print("\n2. Checking if models were saved to disk...")
    models_dir = search_system.models_dir
    
    pubmedbert_path = models_dir / 'pubmedbert'
    reranker_path = models_dir / 'reranker'
    
    print(f"PubMedBERT path: {pubmedbert_path}")
    print(f"PubMedBERT exists: {pubmedbert_path.exists()}")
    if pubmedbert_path.exists():
        print(f"PubMedBERT contents: {list(pubmedbert_path.iterdir())}")
    
    print(f"Reranker path: {reranker_path}")
    print(f"Reranker exists: {reranker_path.exists()}")
    if reranker_path.exists():
        print(f"Reranker contents: {list(reranker_path.iterdir())}")
    
    # Test 2: Load models from cache
    print("\n3. Testing model loading from cache...")
    try:
        pubmedbert_model = search_system.load_model('pubmedbert')
        print(f"✅ PubMedBERT loaded successfully: {type(pubmedbert_model)}")
    except Exception as e:
        print(f"❌ Error loading PubMedBERT: {e}")
    
    try:
        reranker_model = search_system.load_model('reranker')
        print(f"✅ Reranker loaded successfully: {type(reranker_model)}")
    except Exception as e:
        print(f"❌ Error loading Reranker: {e}")
    
    # Test 3: Build and test collection
    print("\n4. Testing collection building...")
    test_sentences = [
        "Chronic Kidney Disease Stage 1 with normal GFR",
        "Chronic Kidney Disease Stage 2 with mild decrease in GFR",
        "Chronic Kidney Disease Stage 4 with severe decrease in GFR",
        "Type 2 Diabetes Mellitus with diabetic nephropathy",
        "Heart Failure with reduced ejection fraction"
    ]
    
    search_system.build_collection(
        sentences=test_sentences,
        collection_name="test_collection",
        model_key="pubmedbert",
        force_rebuild=True
    )
    
    # Test 4: Test search functionality
    print("\n5. Testing search functionality...")
    
    # Test bi-encoder search
    print("\n🔍 Testing bi-encoder search...")
    try:
        results_bi = search_system.search(
            query="CKD Stage 4 treatment",
            collection_name="test_collection",
            method="bi-encoder",
            top_k=3,
            show_details=False
        )
        print("✅ Bi-encoder search successful!")
        for i, (sentence, score) in enumerate(results_bi, 1):
            print(f"  {i}. {score:.4f} - {sentence}")
    except Exception as e:
        print(f"❌ Bi-encoder search failed: {e}")
    
    # Test hybrid search
    print("\n🎯 Testing hybrid search...")
    try:
        results_hybrid = search_system.search(
            query="CKD Stage 4 treatment",
            collection_name="test_collection",
            method="hybrid",
            top_k=3,
            show_details=False
        )
        print("✅ Hybrid search successful!")
        for i, (sentence, score) in enumerate(results_hybrid, 1):
            print(f"  {i}. {score:.4f} - {sentence}")
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
    
    # Test 5: Verify offline functionality
    print("\n6. Testing offline functionality...")
    print("Clearing loaded models from memory...")
    search_system.loaded_models.clear()
    
    print("Loading models from cache again...")
    try:
        pubmedbert_model = search_system.load_model('pubmedbert')
        reranker_model = search_system.load_model('reranker')
        print("✅ Models loaded from cache successfully!")
    except Exception as e:
        print(f"❌ Error loading from cache: {e}")
    
    # Show system status
    print("\n7. System status:")
    search_system.list_models()
    search_system.list_collections()
    
    print("\n🎉 Offline model testing complete!")

if __name__ == "__main__":
    test_offline_models() 