import os
import pickle
import faiss
import numpy as np
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from tools.reranker_tool import gpt_rerank
from utils.preprocessing import preprocess_text


class ProductSearchTool:
    def __init__(self, bm25_index_file, bm25_corpus_file, faiss_index_file, faiss_corpus_file):
        # Load BM25
        with open(bm25_index_file, "rb") as f:
            self.bm25_index = pickle.load(f)
        with open(bm25_corpus_file, "rb") as f:
            self.bm25_corpus = pickle.load(f)

        # Load FAISS
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.faiss_index = faiss.read_index(faiss_index_file)
        with open(faiss_corpus_file, "rb") as f:
            self.faiss_corpus = pickle.load(f)

    def generate_query_variations(self, query):
        pattern = re.compile(r"\b(ux|cx|bx)[\s\-]?(\d{1,2})\b", re.IGNORECASE)
        variations = [query]
        modified = pattern.sub(lambda m: f"{m.group(1).lower()}-{m.group(2).zfill(2)}", query)
        if modified != query:
            variations.append(modified)
        return variations

    def bm25_search(self, query, top_k=20):
        variations = self.generate_query_variations(query)
        results = {}
        for var in variations:
            tokens = preprocess_text(var)
            scores = self.bm25_index.get_scores(tokens)
            for i in np.argsort(scores)[::-1][:top_k]:
                chunk = self.bm25_corpus[i]
                if chunk not in results:
                    results[chunk] = scores[i]
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [text for text, _ in ranked]

    def faiss_search(self, query, top_k=20):
        query_embedding = self.embeddings.embed_query(query)
        D, I = self.faiss_index.search(np.array([query_embedding]).astype("float32"), top_k)
        return [self.faiss_corpus[i] for i in I[0] if i < len(self.faiss_corpus)]

    def run(self, query):
        print(f"\n🔍 [PRODUCT TOOL] Hybrid search for query: '{query}'")

        bm25_results = self.bm25_search(query, top_k=20)
        faiss_results = self.faiss_search(query, top_k=20)

        # Combine and deduplicate
        combined = list(dict.fromkeys(bm25_results + faiss_results))

        # GPT rerank for final relevance
        return gpt_rerank(query, combined, top_k = 6)
