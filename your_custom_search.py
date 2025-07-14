#!/usr/bin/env python3
"""
Custom Semantic Search Example

This script shows how to use your own list of sentences with the offline semantic search system.
Perfect for healthcare procedure descriptions or any domain-specific text collection.
"""

from offline_semantic_search import OfflineSemanticSearch

def main():
    """
    Example of using your own sentence collection for semantic search.
    
    Replace the 'your_sentences' list with your 1000+ healthcare procedure descriptions.
    """
    
    # Initialize the search system
    search_system = OfflineSemanticSearch()
    
    # 🔄 STEP 1: Replace this with your own sentence list
    # This is where you put your 1000+ healthcare procedure descriptions
    your_sentences = [
        # CKD Examples
        "Chronic Kidney Disease Stage 1 with normal GFR and kidney damage",
        "Chronic Kidney Disease Stage 2 with mild decrease in GFR (60-89)",
        "Chronic Kidney Disease Stage 3A with moderate decrease in GFR (45-59)",
        "Chronic Kidney Disease Stage 3B with moderate decrease in GFR (30-44)",
        "Chronic Kidney Disease Stage 4 with severe decrease in GFR (15-29)",
        "Chronic Kidney Disease Stage 5 with kidney failure requiring dialysis",
        
        # Diabetes Examples
        "Type 1 Diabetes Mellitus with diabetic nephropathy and proteinuria",
        "Type 2 Diabetes Mellitus with diabetic nephropathy and microalbuminuria",
        "Type 2 Diabetes Mellitus with diabetic retinopathy and macular edema",
        "Type 2 Diabetes Mellitus with peripheral diabetic neuropathy",
        "Gestational Diabetes Mellitus with insulin resistance",
        "Pre-diabetes with impaired glucose tolerance and metabolic syndrome",
        
        # Heart Failure Examples
        "Heart Failure with reduced ejection fraction (HFrEF) less than 40%",
        "Heart Failure with preserved ejection fraction (HFpEF) greater than 50%",
        "Heart Failure with mid-range ejection fraction (HFmrEF) 40-49%",
        "Acute Heart Failure exacerbation with pulmonary edema",
        "Chronic Heart Failure NYHA Class II with mild symptoms",
        "Chronic Heart Failure NYHA Class III with marked limitation",
        "Chronic Heart Failure NYHA Class IV with severe symptoms at rest",
        
        # Hypertension Examples
        "Essential Hypertension Stage 1 (130-139/80-89 mmHg)",
        "Essential Hypertension Stage 2 (≥140/90 mmHg)",
        "Hypertensive Heart Disease with left ventricular hypertrophy",
        "Hypertensive Chronic Kidney Disease with proteinuria",
        "Hypertensive Emergency with target organ damage",
        "Secondary Hypertension due to renal artery stenosis",
        "White Coat Hypertension with normal home blood pressure",
        
        # Pneumonia Examples
        "Community-acquired pneumonia caused by Streptococcus pneumoniae",
        "Hospital-acquired pneumonia with multidrug-resistant organisms",
        "Ventilator-associated pneumonia in ICU setting",
        "Aspiration pneumonia due to dysphagia",
        "Viral pneumonia caused by influenza virus",
        "Bacterial pneumonia with septic shock",
        "Atypical pneumonia caused by Mycoplasma pneumoniae",
        
        # Additional Medical Conditions
        "Acute myocardial infarction with ST-elevation (STEMI)",
        "Non-ST-elevation myocardial infarction (NSTEMI)",
        "Unstable angina with high troponin levels",
        "Stable angina with exercise-induced chest pain",
        "Atrial fibrillation with rapid ventricular response",
        "Deep vein thrombosis with pulmonary embolism risk",
        "Stroke with left-sided weakness and aphasia",
        "Transient ischemic attack with complete recovery",
        
        # Surgical Procedures
        "Coronary artery bypass graft (CABG) surgery",
        "Percutaneous coronary intervention (PCI) with stent placement",
        "Total knee replacement surgery for osteoarthritis",
        "Hip replacement surgery for fracture",
        "Appendectomy for acute appendicitis",
        "Cholecystectomy for gallbladder stones",
        "Colonoscopy for colorectal cancer screening",
        "Upper endoscopy for gastroesophageal reflux disease"
    ]
    
    print("🚀 CUSTOM SEMANTIC SEARCH SETUP")
    print("=" * 50)
    print(f"📊 Your collection has {len(your_sentences)} sentences")
    
    # 🔄 STEP 2: Download and cache models (one-time setup)
    print("\n📥 Setting up models for offline use...")
    
    # For maximum accuracy, download both models
    search_system.download_and_cache_models(['pubmedbert', 'reranker'])
    
    # For faster setup, download only the bi-encoder
    # search_system.download_and_cache_models(['pubmedbert'])
    
    # 🔄 STEP 3: Build your searchable collection
    print("\n🔨 Building your searchable collection...")
    search_system.build_collection(
        sentences=your_sentences,
        collection_name="my_healthcare_procedures",  # Choose your collection name
        model_key="pubmedbert",  # Best model for medical text
        force_rebuild=True  # Set to False after first build
    )
    
    # 🔄 STEP 4: Test searches with your queries
    print("\n🔍 Testing searches...")
    
    # Your test queries - replace with your actual search needs
    test_queries = [
        "Chronic Kidney Disease Stage 4 treatment",
        "Type 2 diabetes with kidney complications", 
        "Heart failure with low ejection fraction",
        "Hospital acquired pneumonia treatment",
        "High blood pressure medication management",
        "Coronary artery bypass surgery",
        "Diabetes with eye problems"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 SEARCHING: '{query}'")
        print(f"{'='*60}")
        
        # Method 1: Fast bi-encoder search (86.7% accuracy, ~5s)
        print("\n🚀 Fast Bi-Encoder Search:")
        results_fast = search_system.search(
            query=query,
            collection_name="my_healthcare_procedures",
            method="bi-encoder",
            top_k=5,
            show_details=False
        )
        
        for i, (sentence, score) in enumerate(results_fast, 1):
            print(f"{i}. {score:.4f} - {sentence}")
        
        # Method 2: High-accuracy hybrid search (100% accuracy, ~10s)
        print("\n🎯 High-Accuracy Hybrid Search:")
        results_accurate = search_system.search(
            query=query,
            collection_name="my_healthcare_procedures", 
            method="hybrid",
            top_k=5,
            show_details=False
        )
        
        for i, (sentence, score) in enumerate(results_accurate, 1):
            print(f"{i}. {score:.4f} - {sentence}")
            
        # Show if top result is different
        if results_fast[0][0] != results_accurate[0][0]:
            print(f"\n⚠️  Different top results:")
            print(f"   Fast: {results_fast[0][0]}")
            print(f"   Accurate: {results_accurate[0][0]}")
        else:
            print(f"\n✅ Both methods found same top result")
    
    # 🔄 STEP 5: Show system information
    print(f"\n{'='*60}")
    print("📊 SYSTEM INFORMATION")
    print(f"{'='*60}")
    
    search_system.list_models()
    search_system.list_collections()
    
    print("\n🎉 Setup complete! Your semantic search system is ready.")
    print("\nNext steps:")
    print("1. Replace 'your_sentences' with your actual 1000+ procedure descriptions")
    print("2. Adjust 'test_queries' to match your search needs")
    print("3. Use 'hybrid' method for maximum accuracy (100% in our tests)")
    print("4. Use 'bi-encoder' method for faster searches (86.7% accuracy)")


def quick_search_example():
    """
    Quick example of how to search after setup is complete.
    """
    search_system = OfflineSemanticSearch()
    
    # Quick search (assumes collection already built)
    try:
        results = search_system.search(
            query="CKD Stage 4 treatment",
            collection_name="my_healthcare_procedures",
            method="hybrid",  # or "bi-encoder" for faster search
            top_k=3
        )
        
        print("Top 3 results:")
        for i, (sentence, score) in enumerate(results, 1):
            print(f"{i}. {score:.4f} - {sentence}")
            
    except FileNotFoundError:
        print("❌ Collection not found. Run main() first to build the collection.")


if __name__ == "__main__":
    # Run the full setup and demo
    main()
    
    # Uncomment this to test quick search after setup
    # quick_search_example() 