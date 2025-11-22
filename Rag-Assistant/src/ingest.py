"""
Ingestion Pipeline: Load PDF → Chunk → Embed → Index with FAISS

As per assignment requirements:
- Extract text per page (preserve page numbers)
- Chunk size ~ 250-400 words, 20-40% overlap
- Store metadata: page, section_hint, chunk_id
- Create embeddings with all-MiniLM-L6-v2
- Build FAISS index and persist to disk
"""

import sys
import os
from typing import Dict, List, Tuple  # Add Tuple here
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import utils


def extract_text_from_pdf(pdf_path: str) -> Dict[int, str]:
    """
    Extract text from PDF page by page
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Dictionary mapping page_num -> text
    """
    print(f"\n📖 Extracting text from: {pdf_path}")
    print(f"   PDF: {os.path.basename(pdf_path)}")
    
    page_texts = {}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"   Total pages: {total_pages}\n")
            
            for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
                page_num = i + 1
                text = page.extract_text()
                
                if text:
                    text = utils.clean_text(text)
                    page_texts[page_num] = text
        
        print(f"\n✓ Successfully extracted {len(page_texts)} pages")
        
    except Exception as e:
        print(f"\n❌ Error extracting PDF: {e}")
        sys.exit(1)
    
    return page_texts


def create_chunks(page_texts: Dict[int, str]) -> List[Dict]:
    """
    Create overlapping chunks from page texts
    
    As per assignment:
    - Chunk size: 250-400 words (using 350)
    - Overlap: 20-40% (using 30%)
    - Metadata: page, section_hint, chunk_id
    """
    print(f"\n✂️  Creating chunks...")
    print(f"   Chunk size: {config.CHUNK_SIZE_WORDS} words")
    print(f"   Overlap: {config.OVERLAP_PERCENTAGE * 100}%")
    print(f"   Min chunk size: {config.MIN_CHUNK_SIZE} words\n")
    
    all_chunks = []
    chunk_id = 0
    
    for page_num in tqdm(sorted(page_texts.keys()), desc="Chunking pages"):
        text = page_texts[page_num]
        
        page_chunks = utils.chunk_text_with_overlap(
            text=text,
            page_num=page_num,
            chunk_size_words=config.CHUNK_SIZE_WORDS,
            overlap_pct=config.OVERLAP_PERCENTAGE
        )
        
        # Add chunk IDs
        for chunk in page_chunks:
            chunk['chunk_id'] = chunk_id
            chunk_id += 1
            all_chunks.append(chunk)
    
    print(f"\n✓ Created {len(all_chunks)} chunks")
    print(f"   Average words per chunk: {sum(c['word_count'] for c in all_chunks) / len(all_chunks):.1f}")
    print(f"   Pages covered: {len(set(c['page'] for c in all_chunks))}")
    
    return all_chunks


def create_embeddings(chunks: List[Dict]) -> Tuple[np.ndarray, SentenceTransformer]:
    """
    Create embeddings for all chunks using Sentence-BERT
    
    As per assignment: all-MiniLM-L6-v2
    """
    print(f"\n🔢 Creating embeddings...")
    print(f"   Model: {config.EMBEDDING_MODEL}")
    print(f"   Embedding dimension: {config.EMBEDDING_DIMENSION}\n")
    
    # Load model
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    # Extract texts
    texts = [chunk['text'] for chunk in chunks]
    
    # Create embeddings with progress bar
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False  # Will normalize when building FAISS index
    )
    
    print(f"\n✓ Created embeddings")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dtype: {embeddings.dtype}")
    
    return embeddings, model


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index for similarity search
    
    Using IndexFlatIP (Inner Product) for cosine similarity after L2 normalization
    As per assignment requirement: top-k by cosine similarity
    """
    print(f"\n🗂️  Building FAISS index...")
    
    # Normalize embeddings for cosine similarity
    # After normalization: inner product = cosine similarity
    faiss.normalize_L2(embeddings)
    print(f"   ✓ Normalized embeddings for cosine similarity")
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product
    
    # Add embeddings
    index.add(embeddings)
    
    print(f"   ✓ Index built")
    print(f"   Dimension: {dimension}")
    print(f"   Total vectors: {index.ntotal}")
    print(f"   Index type: IndexFlatIP (Cosine Similarity)")
    
    return index


def main():
    """
    Main ingestion pipeline
    """
    print("="*80)
    print(" FYP HANDBOOK RAG - INGESTION PIPELINE")
    print(" Assignment: Text-only RAG with all-MiniLM-L6-v2 + FAISS")
    print("="*80)
    
    # Check if PDF exists
    if not os.path.exists(config.HANDBOOK_PDF):
        print(f"\n❌ Error: PDF not found at {config.HANDBOOK_PDF}")
        print(f"   Please place the FYP Handbook PDF in: {config.DATA_DIR}/")
        print(f"   Expected filename: handbook.pdf")
        return
    
    # Step 1: Extract text from PDF (preserve page numbers)
    page_texts = extract_text_from_pdf(config.HANDBOOK_PDF)
    
    if not page_texts:
        print("❌ No text extracted from PDF. Exiting.")
        return
    
    # Step 2: Create chunks (250-400 words, 20-40% overlap)
    chunks = create_chunks(page_texts)
    
    if not chunks:
        print("❌ No chunks created. Exiting.")
        return
    
    # Save chunks to JSON (human-readable backup)
    utils.save_chunks(chunks, config.CHUNKS_JSON)
    
    # Step 3: Create embeddings (all-MiniLM-L6-v2)
    embeddings, model = create_embeddings(chunks)
    
    # Step 4: Build FAISS index (cosine similarity)
    index = build_faiss_index(embeddings)
    
    # Step 5: Persist to disk
    print(f"\n💾 Saving to disk...")
    utils.save_embeddings(
        index=index,
        metadata=chunks,
        index_path=config.FAISS_INDEX,
        meta_path=config.META_PKL,
        model_name=config.EMBEDDING_MODEL
    )
    
    # Summary
    print("\n" + "="*80)
    print(" ✅ INGESTION COMPLETE!")
    print("="*80)
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  FAISS index size: {index.ntotal} vectors")
    print(f"  Chunk size: {config.CHUNK_SIZE_WORDS} words")
    print(f"  Overlap: {config.OVERLAP_PERCENTAGE * 100}%")
    print(f"  Model: {config.EMBEDDING_MODEL.split('/')[-1]}")
    print("="*80)
    print("\n📝 Next steps:")
    print("   - CLI: python src/ask.py")
    print("   - UI:  streamlit run src/app.py")
    print()


if __name__ == "__main__":
    main()