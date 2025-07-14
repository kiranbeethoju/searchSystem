from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
import time

def test_bi_encoder(model_name, query, procedures):
    """Test bi-encoder approach and return results with timing"""
    start_time = time.time()
    
    model = SentenceTransformer(model_name)
    procedure_embeddings = model.encode(procedures, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    similarities = util.cos_sim(query_embedding, procedure_embeddings)
    
    scored_procedures = []
    for i in range(len(procedures)):
        scored_procedures.append((procedures[i], similarities[0][i].item()))
    
    scored_procedures.sort(key=lambda x: x[1], reverse=True)
    
    end_time = time.time()
    return scored_procedures, end_time - start_time

def test_hybrid_reranker(retriever_model, reranker_model, query, procedures, top_k=5):
    """Test hybrid reranking approach and return results with timing"""
    start_time = time.time()
    
    # Stage 1: Retrieval
    retriever = SentenceTransformer(retriever_model)
    procedure_embeddings = retriever.encode(procedures, convert_to_tensor=True)
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(query_embedding, procedure_embeddings)[0]
    top_results_indices = np.argpartition(cos_scores, -top_k)[-top_k:]
    
    # Stage 2: Reranking
    reranker = CrossEncoder(reranker_model)
    reranker_pairs = []
    for idx in top_results_indices:
        reranker_pairs.append([query, procedures[idx]])
    
    rerank_scores = reranker.predict(reranker_pairs)
    
    reranked_results = []
    for i in range(len(reranker_pairs)):
        reranked_results.append((procedures[top_results_indices[i]], rerank_scores[i]))
    
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    
    end_time = time.time()
    return reranked_results, end_time - start_time

def run_comprehensive_test():
    """Run comprehensive tests with challenging medical diagnoses"""
    
    # Comprehensive list of similar, challenging medical diagnoses
    medical_procedures = [
        # Chronic Kidney Disease Stages (Very Similar)
        "Chronic Kidney Disease Stage 1 with normal GFR",
        "Chronic Kidney Disease Stage 2 with mild decrease in GFR",
        "Chronic Kidney Disease Stage 3A with moderate decrease in GFR",
        "Chronic Kidney Disease Stage 3B with moderate decrease in GFR", 
        "Chronic Kidney Disease Stage 4 with severe decrease in GFR",
        "Chronic Kidney Disease Stage 5 requiring dialysis",
        
        # Hypertension Classifications (Similar but distinct)
        "Essential Hypertension Stage 1",
        "Essential Hypertension Stage 2", 
        "Hypertensive Heart Disease with heart failure",
        "Hypertensive Chronic Kidney Disease Stage 3",
        "Hypertensive Emergency with target organ damage",
        "Secondary Hypertension due to renal disease",
        
        # Diabetes Types and Complications (Overlapping terms)
        "Type 1 Diabetes Mellitus with diabetic nephropathy",
        "Type 2 Diabetes Mellitus with diabetic nephropathy", 
        "Type 2 Diabetes Mellitus with diabetic retinopathy",
        "Type 2 Diabetes Mellitus with peripheral neuropathy",
        "Gestational Diabetes Mellitus",
        "Pre-diabetes with impaired glucose tolerance",
        
        # Heart Failure Classifications (Similar presentations)
        "Heart Failure with reduced ejection fraction",
        "Heart Failure with preserved ejection fraction", 
        "Acute Heart Failure exacerbation",
        "Chronic Heart Failure NYHA Class II",
        "Chronic Heart Failure NYHA Class III",
        "Congestive Heart Failure with pulmonary edema",
        
        # Pneumonia Types (Similar symptoms, different causes)
        "Community-acquired pneumonia",
        "Hospital-acquired pneumonia",
        "Ventilator-associated pneumonia",
        "Aspiration pneumonia",
        "Viral pneumonia",
        "Bacterial pneumonia",
        
        # Unrelated conditions for contrast
        "Acute appendicitis with perforation",
        "Total knee replacement surgery",
        "Cataract extraction with lens implantation",
        "Routine colonoscopy screening"
    ]
    
    # Test queries that should match specific conditions
    test_queries = [
        ("Chronic Kidney Disease Stage 4 treatment", "CKD Stage 4"),
        ("Type 2 Diabetes with kidney complications", "T2DM with nephropathy"),
        ("Heart Failure with low ejection fraction", "HFrEF"),
        ("Hospital acquired lung infection", "HAP"),
        ("High blood pressure stage 2", "HTN Stage 2")
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE MEDICAL DIAGNOSIS SIMILARITY TESTING")
    print("=" * 80)
    print(f"Testing with {len(medical_procedures)} medical procedures")
    print(f"Running {len(test_queries)} different queries")
    print("=" * 80)
    
    # Model configurations
    models_config = {
        "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
        "SapBERT": "cambridgeltl/SapBERT-UMLS-2020AB",
        "PubMedBERT": "NeuML/pubmedbert-base-embeddings"
    }
    
    retriever_model = "NeuML/pubmedbert-base-embeddings"
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    results_summary = []
    
    for query, query_desc in test_queries:
        print(f"\n{'='*60}")
        print(f"TESTING QUERY: {query}")
        print(f"Expected to match: {query_desc}")
        print(f"{'='*60}")
        
        # Test each bi-encoder model
        for model_name, model_path in models_config.items():
            print(f"\n--- {model_name} (Bi-Encoder) ---")
            try:
                results, exec_time = test_bi_encoder(model_path, query, medical_procedures)
                
                print(f"Execution Time: {exec_time:.2f} seconds")
                print("Top 5 Results:")
                for i, (procedure, score) in enumerate(results[:5]):
                    print(f"{i+1}. {score:.4f} - {procedure}")
                
                # Check if the expected match is in top 3
                top_3_procedures = [proc for proc, _ in results[:3]]
                expected_found = any(query_desc.lower() in proc.lower() for proc in top_3_procedures)
                
                results_summary.append({
                    'query': query_desc,
                    'model': model_name,
                    'approach': 'Bi-Encoder',
                    'top_score': results[0][1],
                    'expected_in_top3': expected_found,
                    'execution_time': exec_time
                })
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        # Test hybrid reranking
        print(f"\n--- Hybrid Reranking (PubMedBERT + Cross-Encoder) ---")
        try:
            results, exec_time = test_hybrid_reranker(retriever_model, reranker_model, query, medical_procedures)
            
            print(f"Execution Time: {exec_time:.2f} seconds")
            print("Top 5 Results:")
            for i, (procedure, score) in enumerate(results[:5]):
                print(f"{i+1}. {score:.4f} - {procedure}")
            
            # Check if the expected match is in top 3
            top_3_procedures = [proc for proc, _ in results[:3]]
            expected_found = any(query_desc.lower() in proc.lower() for proc in top_3_procedures)
            
            results_summary.append({
                'query': query_desc,
                'model': 'PubMedBERT + CrossEncoder',
                'approach': 'Hybrid Reranking',
                'top_score': results[0][1],
                'expected_in_top3': expected_found,
                'execution_time': exec_time
            })
            
        except Exception as e:
            print(f"Error with Hybrid Reranking: {e}")
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Query':<25} {'Model':<25} {'Approach':<15} {'Top Score':<10} {'Expected in Top 3':<18} {'Time (s)':<10}")
    print("-" * 103)
    
    for result in results_summary:
        print(f"{result['query']:<25} {result['model']:<25} {result['approach']:<15} {result['top_score']:<10.4f} {str(result['expected_in_top3']):<18} {result['execution_time']:<10.2f}")
    
    # Calculate accuracy statistics
    print(f"\n{'='*60}")
    print("ACCURACY ANALYSIS")
    print(f"{'='*60}")
    
    approaches = {}
    for result in results_summary:
        approach_key = f"{result['approach']}"
        if approach_key not in approaches:
            approaches[approach_key] = {'correct': 0, 'total': 0, 'avg_time': 0}
        
        approaches[approach_key]['total'] += 1
        approaches[approach_key]['avg_time'] += result['execution_time']
        if result['expected_in_top3']:
            approaches[approach_key]['correct'] += 1
    
    for approach, stats in approaches.items():
        accuracy = (stats['correct'] / stats['total']) * 100
        avg_time = stats['avg_time'] / stats['total']
        print(f"{approach:<20}: {accuracy:.1f}% accuracy ({stats['correct']}/{stats['total']}), Avg Time: {avg_time:.2f}s")

if __name__ == "__main__":
    run_comprehensive_test() 