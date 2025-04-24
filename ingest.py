# ingest.py

import os
import pickle
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document
from utils.preprocessing import preprocess_text
from rank_bm25 import BM25Okapi
import numpy as np

# Paths to source data
PRODUCTS_FILE = "data/shopify_active1.txt"
BLOGS_FILE = "data/shopify_blogs.txt"
POLICIES_FILE = "data/shopify_policies.txt"

os.makedirs("vectorstore", exist_ok=True)

embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# -------- Load text content -------- #
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

products_data = read_file(PRODUCTS_FILE)
blogs_data = read_file(BLOGS_FILE)
policies_data = read_file(POLICIES_FILE)

# -------- Split into chunks -------- #
product_splitter = CharacterTextSplitter(separator="==================================================", chunk_size=1000)
product_chunks = product_splitter.split_text(products_data)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
blog_chunks = splitter.split_text(blogs_data)
policy_chunks = splitter.split_text(policies_data)

# -------- Save product BM25 -------- #
print("🔹 Building BM25 for products...")
tokenized_corpus = [preprocess_text(chunk) for chunk in product_chunks]
bm25_index = BM25Okapi(tokenized_corpus)

with open("vectorstore/bm25_index.pkl", "wb") as f:
    pickle.dump(bm25_index, f)
with open("vectorstore/bm25_corpus.pkl", "wb") as f:
    pickle.dump(product_chunks, f)
print("✅ Saved BM25 index and corpus.")

# -------- Save FAISS for products -------- #
print("🔹 Embedding product chunks...")
product_embeddings = embedding_model.embed_documents(product_chunks)
product_index = faiss.IndexFlatL2(len(product_embeddings[0]))
product_index.add(np.array(product_embeddings).astype("float32"))

with open("vectorstore/product_chunks.pkl", "wb") as f:
    pickle.dump(product_chunks, f)
faiss.write_index(product_index, "vectorstore/product_index.faiss")
print("✅ Saved FAISS product index.")


# -------- Save FAISS for blogs -------- #
print("🔹 Embedding blog chunks...")
blog_embeddings = embedding_model.embed_documents(blog_chunks)
blog_index = faiss.IndexFlatL2(len(blog_embeddings[0]))
blog_index.add(np.array(blog_embeddings).astype("float32"))

with open("vectorstore/blog_chunks.pkl", "wb") as f:
    pickle.dump(blog_chunks, f)
faiss.write_index(blog_index, "vectorstore/blog_index.faiss")
print("✅ Saved FAISS blog index.")

# -------- Save FAISS for policies -------- #
print("🔹 Embedding policy chunks...")
policy_embeddings = embedding_model.embed_documents(policy_chunks)
policy_index = faiss.IndexFlatL2(len(policy_embeddings[0]))
policy_index.add(np.array(policy_embeddings).astype("float32"))

with open("vectorstore/policy_chunks.pkl", "wb") as f:
    pickle.dump(policy_chunks, f)
faiss.write_index(policy_index, "vectorstore/policy_index.faiss")
print("✅ Saved FAISS policy index.")
