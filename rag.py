import os
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from wikipediaapi import Wikipedia

def load_models() -> Tuple[SentenceTransformer, Wikipedia]:
    """Load and return the SentenceTransformer model and Wikipedia API."""
    print('[INFO] Loading SentenceTransformer model...')
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    
    print('[INFO] Loading Wikipedia API...')
    wiki = Wikipedia('RAGBot/1.0', 'en')
    
    return embedding_model, wiki

def get_wikipedia_content(page_title: str) -> List[str]:
    """Fetch and split Wikipedia content into paragraphs."""
    print(f'[INFO] Fetching content for "{page_title}"...')
    _, wiki = load_models()
    doc = wiki.page(page_title).text
    
    print('[INFO] Splitting paragraphs...')
    return [p for p in doc.split('\n') if p.strip()]

def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    print('[INFO] Generating embeddings...')
    return model.encode(texts, normalize_embeddings=True)

def get_top_similar_paragraphs(query: str, paragraphs: List[str], embeddings: np.ndarray, model: SentenceTransformer, top_k: int = 3) -> List[str]:
    """Get the top-k most similar paragraphs to the query."""
    query_embed = model.encode(query, normalize_embeddings=True)
    similarities = np.dot(embeddings, query_embed)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [paragraphs[i] for i in top_indices]

def get_ai_response(context: str, query: str) -> str:
    """Get AI-generated response based on context and query."""
    prompt = f"""
    Use the following CONTEXT to answer the QUESTION at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    CONTEXT: {context}
    QUESTION: {query}
    """
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def main():
    load_dotenv()  # Load environment variables from .env file
    
    embedding_model, _ = load_models()
    paragraphs = get_wikipedia_content('Hayao_Miyazaki')
    embeddings = generate_embeddings(paragraphs, embedding_model)
    
    print('[INFO] Ready to answer questions!')
    
    while True:
        query = input('[USER] Enter your query (or type "exit" to quit): ').strip()
        if query.lower() == 'exit':
            break
        
        top_paragraphs = get_top_similar_paragraphs(query, paragraphs, embeddings, embedding_model)
        context = '\n'.join(top_paragraphs)
        
        answer = get_ai_response(context, query)
        print(f'[AI] {answer}\n')

if __name__ == "__main__":
    main()