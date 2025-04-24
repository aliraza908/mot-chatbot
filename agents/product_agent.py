from tools.product_tool import ProductSearchTool

# Define paths for BM25 and FAISS components
bm25_path = "vectorstore/bm25_index.pkl"
bm25_corpus_path = "vectorstore/bm25_corpus.pkl"
faiss_index_path = "vectorstore/product_index.faiss"
faiss_corpus_path = "vectorstore/product_chunks.pkl"

# Initialize hybrid product search tool
product_tool = ProductSearchTool(
    bm25_index_file=bm25_path,
    bm25_corpus_file=bm25_corpus_path,
    faiss_index_file=faiss_index_path,
    faiss_corpus_file=faiss_corpus_path
)

def handle_product_query(input_text: str) -> list:
    """
    Handles product-related queries by running hybrid search on input text.
    Input could be a refined query or a specific product code like 'UX-09'.
    """
    print(f"\n🔎 [PRODUCT AGENT] Handling product query: '{input_text}'")
    results = product_tool.run(input_text)
    print(f"✅ [PRODUCT AGENT] Retrieved {len(results)} product chunks.")
    return results
