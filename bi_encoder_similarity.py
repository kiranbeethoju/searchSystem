from sentence_transformers import SentenceTransformer, util

def run_semantic_search(model_name, query, procedures):
    """
    Performs semantic search using a bi-encoder model.

    Args:
        model_name (str): The name of the Sentence Transformer model to use from Hugging Face.
        query (str): The search query.
        procedures (list): A list of procedure descriptions to search against.
    """
    print(f"--- Running Semantic Search with: {model_name} ---")

    # Load the clinical embedding model from Hugging Face
    # The model will be downloaded automatically on first use
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}. Please ensure you have an internet connection and the model name is correct.")
        print(e)
        return

    # Encode the query and all procedure descriptions into embeddings
    print("Encoding procedures and query...")
    procedure_embeddings = model.encode(procedures, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity between the query and all procedures
    # util.cos_sim is a utility from sentence-transformers
    similarities = util.cos_sim(query_embedding, procedure_embeddings)

    # Pair each procedure with its similarity score
    scored_procedures = []
    for i in range(len(procedures)):
        scored_procedures.append((procedures[i], similarities[0][i].item()))

    # Sort the procedures by similarity score in descending order
    scored_procedures.sort(key=lambda x: x[1], reverse=True)

    print(f"\nQuery: '{query}'")
    print("\nSearch Results (Sorted by Similarity):")
    for procedure, score in scored_procedures:
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

    # --- Model 1: ClinicalBERT ---
    # Optimized for clinical concept similarity from notes.
    clinical_bert_model = "emilyalsentzer/Bio_ClinicalBERT"
    run_semantic_search(clinical_bert_model, search_query, procedure_database)

    # --- Model 2: SapBERT ---
    # Trained on UMLS ontologies, excellent for resolving similar-looking medical codes.
    sapbert_model = "cambridgeltl/SapBERT-UMLS-2020AB"
    run_semantic_search(sapbert_model, search_query, procedure_database) 