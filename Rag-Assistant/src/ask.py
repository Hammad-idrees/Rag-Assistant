"""
CLI Interface for FYP Handbook RAG System

As per assignment requirements:
- Retrieve top-k=5 chunks by cosine similarity
- Show retrieved chunks (page refs) 
- Apply similarity threshold (0.25)
- Answer ONLY from context with page citations
- Reply "I don't have that in the handbook" if below threshold
"""

import sys
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import utils


class HandbookAssistant:
    """
    FYP Handbook RAG Assistant
    Retrieval-based system (no LLM generation needed as per assignment)
    """
    
    def __init__(self):
        """Initialize by loading FAISS index and embedding model"""
        print("🔄 Loading system...")
        
        # Load FAISS index and metadata
        print(f"   Loading FAISS index from: {config.FAISS_INDEX}")
        self.index, self.metadata = utils.load_embeddings(
            config.FAISS_INDEX,
            config.META_PKL
        )
        print(f"   ✓ Loaded {self.index.ntotal} vectors")
        
        # Load embedding model
        print(f"   Loading embedding model: {config.EMBEDDING_MODEL.split('/')[-1]}")
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"   ✓ Model ready")
        
        # Load prompt template
        with open(config.PROMPT_TEMPLATE_FILE, 'r') as f:
            self.prompt_template = f.read()
        
        print("\n✅ Handbook Assistant ready!")
        print(f"   Retrieval settings: top-k={config.TOP_K}, threshold={config.SIMILARITY_THRESHOLD}")
        print()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks by cosine similarity
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve (default from config)
        
        Returns:
            List of retrieved chunks with scores
        """
        if top_k is None:
            top_k = config.TOP_K
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build result with metadata
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.metadata[idx].copy()
            chunk['score'] = float(score)
            retrieved.append(chunk)
        
        return retrieved
    
    def check_relevance(self, retrieved_chunks: List[Dict]) -> bool:
        """
        Check if retrieved chunks meet similarity threshold
        As per assignment: threshold = 0.25
        """
        if not retrieved_chunks:
            return False
        
        max_score = max(chunk['score'] for chunk in retrieved_chunks)
        return max_score >= config.SIMILARITY_THRESHOLD
    
    def ask(self, question: str, show_debug: bool = False) -> Dict:
        """
        Main method to ask a question
        
        Process:
        1. Embed query
        2. Retrieve top-k=5 chunks
        3. Check similarity threshold
        4. Format answer with page citations
        5. Log prompt
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1 & 2: Retrieve top-k chunks
        retrieved_chunks = self.retrieve(question)
        
        # Step 3: Check relevance threshold
        if not self.check_relevance(retrieved_chunks):
            return {
                "answer": "I don't have that in the handbook.",
                "sources": [],
                "pages": [],
                "confidence": "low",
                "max_score": 0.0
            }
        
        # Step 4: Format answer from retrieved chunks (with page citations)
        answer = utils.format_retrieved_chunks(retrieved_chunks)
        pages = utils.get_unique_pages(retrieved_chunks)
        
        # Step 5: Log prompt (for deliverable)
        utils.log_prompt(question, retrieved_chunks, answer, config.PROMPT_LOG)
        
        # Prepare result
        result = {
            "answer": answer,
            "sources": retrieved_chunks,
            "pages": pages,
            "confidence": "high" if retrieved_chunks[0]['score'] > 0.5 else "medium",
            "max_score": retrieved_chunks[0]['score']
        }
        
        return result
    
    def display_result(self, question: str, result: Dict, show_sources: bool = True):
        """Display results in a formatted way"""
        print("="*80)
        print("QUESTION:")
        print("="*80)
        print(question)
        print()
        
        print("="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        print()
        
        if result['pages']:
            print(f"📄 Referenced Pages: {', '.join(map(str, result['pages']))}")
            print(f"🎯 Confidence: {result['confidence'].upper()} (max score: {result['max_score']:.3f})")
        
        if show_sources and result['sources']:
            print()
            print("="*80)
            print("RETRIEVED SOURCES (Debug Info):")
            print("="*80)
            print(utils.format_sources_display(result['sources']))
        
        print("="*80)
        print()


def main():
    """CLI interface"""
    
    # Check if index exists
    if not os.path.exists(config.FAISS_INDEX):
        print("❌ Error: FAISS index not found!")
        print(f"   Expected at: {config.FAISS_INDEX}")
        print("\n📝 Please run the ingestion pipeline first:")
        print("   python src/ingest.py")
        return
    
    # Initialize assistant
    try:
        assistant = HandbookAssistant()
    except Exception as e:
        print(f"❌ Error initializing assistant: {e}")
        return
    
    # Check if question provided as command-line argument
    if len(sys.argv) > 1:
        # Single question mode
        question = " ".join(sys.argv[1:])
        
        result = assistant.ask(question)
        assistant.display_result(question, result, show_sources=True)
        
    else:
        # Interactive mode
        print("="*80)
        print(" FYP HANDBOOK ASSISTANT - INTERACTIVE MODE")
        print("="*80)
        print(" Ask questions about the FYP Handbook")
        print(" Type 'quit', 'exit', or 'q' to exit")
        print(" Type 'help' for sample questions")
        print("="*80)
        print()
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    print("\n📝 Sample questions:")
                    print("1. What headings, fonts, and sizes are required in the FYP report?")
                    print("2. What margins and spacing do we use?")
                    print("3. What are the required chapters of a Development FYP report?")
                    print("4. What are the required chapters of an R&D FYP report?")
                    print("5. How should endnotes like 'Ibid.' and 'op. cit.' be used?")
                    print("6. What goes into the Executive Summary and Abstract?")
                    print()
                    continue
                
                print()
                result = assistant.ask(question)
                assistant.display_result(question, result, show_sources=True)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()