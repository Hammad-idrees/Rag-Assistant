"""
Utility functions for the FYP Handbook RAG system
"""

import re
import json
import pickle
import faiss
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import config


def chunk_text_with_overlap(text: str, page_num: int, chunk_size_words: int, 
                             overlap_pct: float) -> List[Dict]:
    """
    Split text into overlapping chunks with metadata
    
    Args:
        text: Input text to chunk
        page_num: Source page number
        chunk_size_words: Target chunk size in words
        overlap_pct: Overlap percentage (0.0 to 1.0)
    
    Returns:
        List of chunk dictionaries with metadata
    """
    # Split into words
    words = text.split()
    
    if len(words) < config.MIN_CHUNK_SIZE:
        return []
    
    chunks = []
    overlap_words = int(chunk_size_words * overlap_pct)
    step = chunk_size_words - overlap_words
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size_words]
        
        if len(chunk_words) < config.MIN_CHUNK_SIZE:
            break
            
        chunk_text = " ".join(chunk_words)
        
        # Extract section hint (first line that looks like a heading)
        section_hint = extract_section_hint(chunk_text)
        
        chunk = {
            "text": chunk_text,
            "page": page_num,
            "section": section_hint,
            "word_count": len(chunk_words)
        }
        chunks.append(chunk)
    
    return chunks


def extract_section_hint(text: str) -> str:
    """
    Extract section heading from chunk text
    Looks for ALL CAPS lines or numbered headings
    """
    lines = text.split('\n')
    
    for line in lines[:3]:  # Check first 3 lines
        line = line.strip()
        if len(line) > 5 and len(line) < 100:
            # Check if line is all uppercase or starts with number
            if line.isupper() or re.match(r'^\d+\.?\s+[A-Z]', line):
                return line
    
    # Return first 50 chars if no heading found
    return text[:50].replace('\n', ' ').strip() + "..."


def clean_text(text: str) -> str:
    """Clean extracted text from PDF"""
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove multiple newlines but preserve paragraph breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # Strip
    return text.strip()


def save_chunks(chunks: List[Dict], filepath: str):
    """Save chunks to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(chunks)} chunks to {filepath}")


def load_chunks(filepath: str) -> List[Dict]:
    """Load chunks from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_embeddings(index: faiss.Index, metadata: List[Dict], 
                    index_path: str, meta_path: str, model_name: str):
    """Save FAISS index and metadata to disk"""
    faiss.write_index(index, index_path)
    
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    with open(config.MODEL_NAME_FILE, 'w') as f:
        f.write(model_name)
    
    print(f"✓ Saved FAISS index to {index_path}")
    print(f"✓ Saved metadata to {meta_path}")


def load_embeddings(index_path: str, meta_path: str) -> Tuple[faiss.Index, List[Dict]]:
    """Load FAISS index and metadata from disk"""
    index = faiss.read_index(index_path)
    
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata


def format_retrieved_chunks(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a readable answer with citations
    This follows the assignment's prompt template structure
    """
    if not chunks:
        return "I don't have that in the handbook."
    
    # Group chunks by page for better organization
    page_groups = {}
    for chunk in chunks:
        page = chunk['page']
        if page not in page_groups:
            page_groups[page] = []
        page_groups[page].append(chunk)
    
    # Build answer from chunks
    answer_parts = []
    
    for page, page_chunks in sorted(page_groups.items()):
        # Combine chunks from same page
        page_text = "\n".join(c['text'] for c in page_chunks)
        answer_parts.append(f"{page_text} (p. {page})")
    
    return "\n\n".join(answer_parts)


def format_sources_display(chunks: List[Dict]) -> str:
    """Format sources for collapsible display in UI"""
    sources = []
    
    for i, chunk in enumerate(chunks, 1):
        source = f"""
Source {i}:
  Page: {chunk['page']}
  Section: {chunk.get('section', 'N/A')}
  Similarity: {chunk.get('score', 0):.3f}
  Preview: {chunk['text'][:200]}...
"""
        sources.append(source)
    
    return "\n".join(sources)


def log_prompt(question: str, retrieved_chunks: List[Dict], 
               answer: str, filepath: str):
    """
    Log prompts and answers to file for deliverable
    As per assignment requirement: "Prompt log (txt)"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format context as per prompt template
    context = "\n\n".join([
        f"[Chunk {i+1} - Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    ])
    
    log_entry = f"""
{'='*80}
Timestamp: {timestamp}
{'='*80}

QUESTION:
{question}

RETRIEVED CONTEXT (Top-{len(retrieved_chunks)} chunks):
{context}

SIMILARITY SCORES:
{', '.join([f"Chunk {i+1}: {c.get('score', 0):.3f}" for i, c in enumerate(retrieved_chunks)])}

ANSWER:
{answer}

PAGE REFERENCES:
Pages: {', '.join(str(c['page']) for c in retrieved_chunks)}

"""
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def get_unique_pages(chunks: List[Dict]) -> List[int]:
    """Extract unique page numbers from chunks"""
    return sorted(list(set(chunk['page'] for chunk in chunks)))