from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np

def run_hybrid_reranking_search(retriever_model_name, reranker_model_name, query, procedures, top_k=3):
    """
    Performs a two-stage search: retrieval with a bi-encoder and
    reranking with a cross-encoder.

    Args:
        retriever_model_name (str): The bi-encoder model for initial retrieval.
        reranker_model_name (str): The cross-encoder model for precision reranking.
        query (str): The search query.
        procedures (list): The list of procedure descriptions.
        top_k (int): The number of candidates to retrieve before reranking.
    """
    print(f"--- Running Hybrid Reranking Search ---")
    print(f"Retriever: {retriever_model_name}")
    print(f"Reranker: {reranker_model_name}")

    # --- Stage 1: Retrieval ---
    # Use a fast bi-encoder to find potentially relevant candidates.
    retriever = SentenceTransformer(retriever_model_name)

    print("\nEncoding all procedures for initial retrieval...")
    procedure_embeddings = retriever.encode(procedures, convert_to_tensor=True)
    query_embedding = retriever.encode(query, convert_to_tensor=True)

    # Find the top_k most similar procedures using cosine similarity
    print(f"Retrieving top {top_k} candidates...")
    cos_scores = util.cos_sim(query_embedding, procedure_embeddings)[0]
    
    # Use numpy.argpartition to efficiently find the top_k indices without a full sort
    top_results_indices = np.argpartition(cos_scores, -top_k)[-top_k:]
    
    print("\nInitial Top Candidates (before reranking):")
    for idx in top_results_indices:
        print(f"{cos_scores[idx]:.4f} - {procedures[idx]}")

    # --- Stage 2: Reranking ---
    # Use a more powerful cross-encoder to re-rank the top candidates.
    reranker = CrossEncoder(reranker_model_name)

    # Create pairs of (query, candidate) for the cross-encoder
    reranker_pairs = []
    for idx in top_results_indices:
        reranker_pairs.append([query, procedures[idx]])

    print("\nReranking with Cross-Encoder for higher precision...")
    # The cross-encoder gives a more accurate score for each pair.
    rerank_scores = reranker.predict(reranker_pairs)

    # Pair the candidates with their new reranked scores
    reranked_results = []
    for i in range(len(reranker_pairs)):
        reranked_results.append((procedures[top_results_indices[i]], rerank_scores[i]))

    # Sort the results by the new reranker score
    reranked_results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n--- Final Results for Query: '{query}' ---")
    print("(Sorted by Cross-Encoder Reranking Score)")
    for procedure, score in reranked_results:
        print(f"{score:.4f} - {procedure}")
    print("-" * 50)

if __name__ == '__main__':
    # Your database of procedure descriptions
    procedure_database = [
        "Chronic Kidney Disease Stage 2 Monitoring",
        "Chronic Kidney Disease Stage 3 Dialysis Prep",
        "Chronic Kidney Disease Stage 4 Hemodialysis",
        "Acute Kidney Injury Treatment",
        "End-Stage Renal Disease with Transplant",
        "Glomerulonephritis Management",
        "Coronary Artery Bypass Graft"
    ]

    # The search query we want to match
    search_query = "Chronic Kidney Disease Stage 4 Treatment"

    # --- Model Definitions ---
    # A biomedical-tuned bi-encoder for fast retrieval
    retriever_model = "NeuML/pubmedbert-base-embeddings"
    # A cross-encoder trained for ranking, providing higher accuracy
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    run_hybrid_reranking_search(retriever_model, reranker_model, search_query, procedure_database) 