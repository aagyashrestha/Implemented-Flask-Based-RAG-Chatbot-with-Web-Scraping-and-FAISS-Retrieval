import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from typing import List

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Scrape text content from a list of webpages
def scrape_webpages(urls: List[str]) -> List[str]:
    texts = []
    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            body_text = soup.get_text(separator="\n", strip=True)
            texts.append(body_text)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return texts

# Embed a list of texts of the url using OpenAI embeddings
def embed_texts(texts: List[str]):
    chunks = [text[:2000] for text in texts]  # truncate to fit token limits - 2000 characters
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunks
    )
    return [embedding.embedding for embedding in response.data]

# Normalize vectors for cosine similarity in FAISS
def normalize(vectors: List[List[float]]):
    array = np.array(vectors).astype("float32") #Converts the list of vectors into a NumPy array (faster and required by FAISS).
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / norms

# Core function: scrape, embed, retrieve, generate answer
def ask_rag_bot_from_urls(question: str, urls: List[str]) -> str:
    docs = scrape_webpages(urls)
    if not docs:
        return "Failed to scrape any content from the provided URLs."

    # Split each document into smaller chunks (around 1000 tokens or ~4000 characters)
    chunks = []
    for doc in docs:
        for i in range(0, len(doc), 4000):
            chunks.append(doc[i:i+4000])

    vectors = embed_texts(chunks)
    vectors = normalize(vectors)  # normalize for cosine-like behavior

    # Create a FAISS index for inner product (works like cosine after normalization)
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    q_vec = embed_texts([question])[0]
    q_vec = normalize([q_vec])

    # Search top 3 most similar chunks
    top_k = 3
    scores, top_indices = index.search(q_vec, top_k)
    top_chunks = [chunks[i] for i in top_indices[0]]

    # Reduce prompt size by only including top relevant chunks
    context = "\n---\n".join(top_chunks)
    if len(context) > 12000:
        context = context[:12000]

    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content
