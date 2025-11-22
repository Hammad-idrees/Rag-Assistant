"""
Streamlit UI for FYP Handbook RAG Assistant

As per assignment requirements:
- One input box
- One "Ask" button
- Answer panel
- Collapsible "Sources (page refs)" list
"""

import sys
import os
import streamlit as st

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.ask import HandbookAssistant
from src import utils


# Page config
st.set_page_config(
    page_title="FYP Handbook Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        white-space: pre-wrap;
    }
    .info-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assistant():
    """Load assistant (cached for performance)"""
    return HandbookAssistant()


def main():
    # Header
    st.markdown('<div class="main-header">📚 FYP Handbook Assistant</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">FAST-NUCES Final Year Project Handbook 2023</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Check if index exists
    if not os.path.exists(config.FAISS_INDEX):
        st.error("❌ **FAISS index not found!**")
        st.info("Please run the ingestion pipeline first: `python src/ingest.py`")
        return
    
    # Load assistant
    try:
        with st.spinner("🔄 Loading models... (first load may take a minute)"):
            assistant = load_assistant()
    except Exception as e:
        st.error(f"❌ Error loading assistant: {e}")
        return
    
    # Sidebar - Configuration Info
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        st.markdown("### Chunking")
        st.markdown(f"""
        - **Chunk size:** {config.CHUNK_SIZE_WORDS} words
        - **Overlap:** {config.OVERLAP_PERCENTAGE * 100}%
        - **Min size:** {config.MIN_CHUNK_SIZE} words
        """)
        
        st.markdown("### Retrieval")
        st.markdown(f"""
        - **Model:** {config.EMBEDDING_MODEL.split('/')[-1]}
        - **Top-K:** {config.TOP_K}
        - **Threshold:** {config.SIMILARITY_THRESHOLD}
        - **Similarity:** Cosine
        """)
        
        st.markdown("---")
        st.header("📝 Sample Questions")
        
        sample_questions = [
            "What headings, fonts, and sizes are required in the FYP report?",
            "What margins and spacing do we use?",
            "What are the required chapters of a Development FYP report?",
            "What are the required chapters of an R&D FYP report?",
            "How should endnotes like 'Ibid.' and 'op. cit.' be used?",
            "What goes into the Executive Summary and Abstract?"
        ]
        
        for i, q in enumerate(sample_questions, 1):
            if st.button(f"Q{i}: {q[:50]}...", key=f"sample_{i}", use_container_width=True):
                st.session_state.question = q
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.85rem; color: gray;'>
        RAG System using<br>
        Sentence-BERT + FAISS<br><br>
        FAST-NUCES FYP 2023
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    st.markdown("### 💬 Ask Your Question")
    
    # Initialize session state
    if 'question' not in st.session_state:
        st.session_state.question = ""
    
    # Question input
    question = st.text_area(
        "Enter your question about the FYP Handbook:",
        value=st.session_state.question,
        height=100,
        placeholder="e.g., What are the formatting requirements for the FYP report?",
        key="question_input"
    )
    
    # Buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.question = ""
        st.rerun()
    
    # Process question
    if ask_button and question.strip():
        with st.spinner("🔍 Searching handbook..."):
            try:
                result = assistant.ask(question.strip())
                
                # Display answer
                st.markdown("---")
                st.markdown("### 📝 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', 
                           unsafe_allow_html=True)
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    if result['pages']:
                        st.info(f"📄 **Pages:** {', '.join(map(str, result['pages']))}")
                    else:
                        st.warning("📄 **Pages:** None")
                
                with col2:
                    confidence_emoji = {
                        "high": "🟢",
                        "medium": "🟡",
                        "low": "🔴"
                    }
                    emoji = confidence_emoji.get(result['confidence'], '⚪')
                    st.info(f"{emoji} **Confidence:** {result['confidence'].title()}")
                
                with col3:
                    st.info(f"🎯 **Max Score:** {result['max_score']:.3f}")
                
                # Display sources in expander (as per assignment: collapsible)
                if result['sources']:
                    with st.expander("📚 View Retrieved Sources (Page References)", expanded=False):
                        st.markdown("**Retrieved Chunks (Top-5 by Cosine Similarity):**")
                        
                        for i, chunk in enumerate(result['sources'], 1):
                            st.markdown(f"#### Source {i}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Page", chunk['page'])
                            with col2:
                                st.metric("Similarity", f"{chunk['score']:.3f}")
                            with col3:
                                st.metric("Words", chunk['word_count'])
                            
                            st.markdown(f"**Section:** {chunk.get('section', 'N/A')}")
                            
                            st.text_area(
                                f"Chunk {i} Text",
                                value=chunk['text'],
                                height=150,
                                key=f"chunk_{i}",
                                disabled=True
                            )
                            
                            if i < len(result['sources']):
                                st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
        Built for FAST-NUCES FYP Assignment | 
        Retrieval-Augmented Generation (RAG) System<br>
        Using: sentence-transformers/all-MiniLM-L6-v2 + FAISS
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()