import os
import google.generativeai as genai
import time
from typing import List, Dict, Optional
from vector_store import VectorStore


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.vector_store = vector_store
        self.chat_history: List[Dict] = []

    def _generate_with_retry(self, prompt: str, retries: int = 5, base_delay: float = 2.0):
        """Generate content with retry logic for rate limits."""
        for i in range(retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                    )
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                # Check for rate limit errors (429)
                if "429" in error_str or "rate_limit_exceeded" in error_str.lower() or "quota" in error_str.lower():
                    if i < retries - 1:
                        sleep_time = base_delay * (2 ** i) 
                        print(f"Rate limit hit. Retrying in {sleep_time:.2f}s... (Attempt {i+1}/{retries})")
                        time.sleep(sleep_time)
                        continue
                raise e

    def answer(self, question: str, top_k: int = 5) -> Dict:
        """Retrieve relevant chunks and generate an answer."""
        # Retrieval
        retrieved_chunks = self.vector_store.search(question, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                "answer": "I could not find relevant information in the uploaded document to answer your question.",
                "citations": [],
                "question": question
            }
        
        # Build context from retrieved chunks
        context_parts = []
        citations = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"[Source {i+1} - Page {chunk['page_num']}]\n{chunk['text']}"
            )
            citations.append({
                "source_num": i + 1,
                "page_num": chunk["page_num"],
                "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
                "score": round(chunk["score"], 4)
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build conversation history string
        history_str = ""
        if self.chat_history:
            history_parts = []
            for h in self.chat_history[-4:]:  # last 2 Q&A pairs
                history_parts.append(f"User: {h['question']}\nAssistant: {h['answer']}")
            history_str = "\n\n".join(history_parts) + "\n\n"
        
        # Prompt
        prompt = f"""You are an intelligent document assistant. Answer the user's question **strictly based on the provided document excerpts**. 
If the answer is not in the context, say so clearly. Be concise, accurate, and cite page numbers when relevant.

## Document Context:
{context}

## Conversation History:
{history_str if history_str else "No previous conversation."}

## Current Question:
{question}

## Answer:"""
        
        try:
            answer_text = self._generate_with_retry(prompt)
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error generating the answer. Please try again in a moment. (Error: {str(e)})",
                "citations": citations,
                "question": question
            }
        
        # Update chat history
        self.chat_history.append({
            "question": question,
            "answer": answer_text,
            "citations": citations
        })
        
        return {
            "answer": answer_text,
            "citations": citations,
            "question": question
        }

    def summarize(self) -> str:
        """Generate a document summary from top-level chunks."""
        if not self.vector_store.chunks:
            return "No document loaded."
        
        # Sample chunks evenly across the document
        chunks = self.vector_store.chunks
        step = max(1, len(chunks) // 10)
        sampled = chunks[::step][:10]
        
        context = "\n\n".join([f"[Page {c['page_num']}] {c['text']}" for c in sampled])
        
        prompt = f"""You are a document summarizer. Based on the following excerpts from a document, write a comprehensive summary (3-5 paragraphs). 
Cover the main topics, key points, and important conclusions.

## Document Excerpts:
{context}

## Summary:"""
        
        try:
            return self._generate_with_retry(prompt)
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return "Summary unavailable due to high API load. You can still ask questions about the document."

    def clear_history(self):
        self.chat_history = []
