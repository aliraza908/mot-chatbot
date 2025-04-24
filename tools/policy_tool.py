# tools/policy_tool.py

import faiss
import os
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np


class PolicySearchTool:
    def __init__(self, faiss_index_path="vectorstore/policy_index.faiss", store_path="vectorstore/policy_chunks.pkl"):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.index = faiss.read_index(faiss_index_path)
        with open(store_path, "rb") as f:
            self.raw_chunks = pickle.load(f)

    def run(self, query, top_k=10):
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array([query_embedding]).astype("float32")  # ✅ FIX
        D, I = self.index.search(query_embedding, top_k)
        return [self.raw_chunks[i] for i in I[0] if i < len(self.raw_chunks)]