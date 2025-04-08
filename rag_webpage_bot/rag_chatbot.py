#rag_chatbot.py
import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from typing import List

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Limit to last 7 user-bot exchanges (14 messages)
MAX_MEMORY = 7

# In-memory chat history store per user
memory_store = {}

def get_chat_history(user_id: str) -> List[dict]:
    return memory_store.get(user_id, [])

def save_chat_history(user_id: str, conversation: List[dict]):
    memory_store[user_id] = conversation[-MAX_MEMORY * 2:]

# Scrape and clean text from provided URLs
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

# Embed text chunks using OpenAI embeddings
def embed_texts(texts: List[str]):
    chunks = [text[:2000] for text in texts]
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=chunks
    )
    return [embedding.embedding for embedding in response.data]

# Normalize vectors for cosine similarity
def normalize(vectors: List[List[float]]):
    array = np.array(vectors).astype("float32")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / norms

# Main RAG chatbot function
def ask_rag_bot_from_urls(question: str, urls: List[str], user_id: str) -> str:
    docs = scrape_webpages(urls)
    if not docs:
        return "Failed to scrape any content from the provided URLs."

    # Split scraped docs into chunks
    chunks = []
    for doc in docs:
        for i in range(0, len(doc), 4000):
            chunks.append(doc[i:i+4000])

    # Get embeddings
    vectors = normalize(embed_texts(chunks))
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Embed and search question
    q_vec = normalize([embed_texts([question])[0]])
    scores, top_indices = index.search(q_vec, 3)
    top_chunks = [chunks[i] for i in top_indices[0]]
    context = "\n---\n".join(top_chunks)[:12000]

    # Build and update chat history
    history = get_chat_history(user_id)
    history.append({"role": "user", "content": question})
    history = history[-MAX_MEMORY * 2:]

    # Create prompt using history and context
    prompt = [
        *history,
        {"role": "user", "content": f"Use the following context to answer:\n\n{context}\n\nQuestion: {question}"}
    ]

    # Get bot response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()

    if any(x in answer.lower() for x in ["no mention", "not clear", "no information"]):
        answer += "\n\n🤔 I couldn’t find specific details from the provided content. Could you give me more context?"

    # Update memory with bot response
    history.append({"role": "assistant", "content": answer})
    save_chat_history(user_id, history)

    # Format chat log for output (plain text)
    trimmed_history = history[-MAX_MEMORY * 2:]
    raw_chat_log = ""
    for i in range(0, len(trimmed_history), 2):
        user_msg = trimmed_history[i]["content"] if i < len(trimmed_history) else ""
        bot_msg = trimmed_history[i + 1]["content"] if i + 1 < len(trimmed_history) else ""
        raw_chat_log += f"{{user: {user_msg}}}\n"
        raw_chat_log += f"{{bot: {bot_msg}}}\n"

    return raw_chat_log
