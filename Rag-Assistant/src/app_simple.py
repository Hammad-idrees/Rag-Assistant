"""
Streamlit UI for FYP Handbook RAG System
Beautiful and interactive version - Enhanced Design
"""

import streamlit as st
import sys
import os
import numpy as np
import faiss
from typing import List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils import load_embeddings, format_retrieved_chunks, format_sources_display, log_prompt


class SimpleHandbookAssistant:
    """
    Simplified assistant that uses the pre-computed FAISS index
    without needing to load sentence-transformers at runtime
    """
    
    def __init__(self):
        """Initialize with pre-built index only"""
        with st.spinner("📚 Loading FYP Handbook knowledge base..."):
            # Load FAISS index and metadata
            self.index, self.metadata = load_embeddings(config.FAISS_INDEX, config.META_PKL)
    
    def simple_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Simple retrieval using text matching
        """
        query_lower = query.lower()
        
        scored_chunks = []
        for chunk in self.metadata:
            chunk_text = chunk['text'].lower()
            
            # Simple word overlap scoring
            query_words = set(query_lower.split())
            chunk_words = set(chunk_text.split())
            common_words = query_words.intersection(chunk_words)
            score = len(common_words) / len(query_words) if query_words else 0
            
            if score > 0:
                chunk_with_score = chunk.copy()
                chunk_with_score['score'] = score
                scored_chunks.append((chunk_with_score, score))
        
        # Sort by score and return top-k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks[:top_k]]
    
    def answer_question(self, question: str):
        """Answer question using simple retrieval"""
        # Retrieve relevant chunks
        chunks = self.simple_retrieve(question, config.TOP_K)
        
        if not chunks:
            return "I don't have that in the handbook.", [], 0.0
        
        # Apply similarity threshold
        max_score = chunks[0]['score']
        if max_score < config.SIMILARITY_THRESHOLD:
            return "I don't have that in the handbook.", [], 0.0
        
        # Format answer
        answer = format_retrieved_chunks(chunks)
        
        # Log the interaction
        log_prompt(question, chunks, answer, config.PROMPT_LOG)
        
        return answer, chunks, max_score


def init_session_state():
    """Initialize session state variables"""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = SimpleHandbookAssistant()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_question' not in st.session_state:
        st.session_state.last_question = ""


def display_chat_message(role: str, content: str, pages: List[int] = None, confidence: float = 0.0):
    """Display a chat message with proper formatting"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {content}")
    else:
        with st.chat_message("assistant"):
            # Confidence indicator
            if confidence > 0:
                confidence_color = "🟢" if confidence > 0.5 else "🟡" if confidence > 0.25 else "🔴"
                st.caption(f"{confidence_color} Confidence: {confidence:.1%}")
            
            # Page references
            if pages:
                st.caption(f"📖 Referenced pages: {', '.join(map(str, pages))}")
            
            # Answer content
            st.markdown(content)


def display_source_details(sources: List[Dict]):
    """Display source details in an expandable section"""
    with st.expander("🔍 **View Source Details**", expanded=False):
        st.markdown("### Retrieved Chunks")
        
        for i, chunk in enumerate(sources, 1):
            # Create columns for source info
            col1, col2, col3 = st.columns([1, 2, 7])
            
            with col1:
                st.metric(
                    label=f"Source {i}",
                    value=f"P.{chunk['page']}",
                    delta=f"{chunk['score']:.3f}"
                )
            
            with col2:
                confidence = "HIGH" if chunk['score'] > 0.5 else "MEDIUM" if chunk['score'] > 0.25 else "LOW"
                st.info(f"**{confidence}**")
            
            with col3:
                # Display chunk preview
                preview = chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
                st.text_area(
                    f"Content preview",
                    value=preview,
                    height=80,
                    key=f"chunk_{i}",
                    disabled=True
                )
            
            st.markdown("---")


def display_quick_questions():
    """Display quick question buttons with clear text"""
    st.markdown("### 💡 Quick Questions")
    
    # Updated to show full question text clearly
    quick_questions = [
        ("📏", "Margins & Spacing", "What margins and spacing do we use?"),
        ("🔤", "Fonts & Styles", "What headings, fonts, and sizes are required?"),
        ("📑", "Report Structure", "What are the required chapters of a Development FYP report?"),
        ("📚", "Citations", "How should endnotes like 'Ibid.' and 'op. cit.' be used?"),
        ("📄", "Abstract & Summary", "What goes into the Executive Summary and Abstract?"),
        ("🔬", "R&D Reports", "What are the required chapters of an R&D-based FYP report?")
    ]
    
    # Display each question button
    for idx, (icon, title, question) in enumerate(quick_questions):
        if st.button(
            f"{icon} **{title}**",
            use_container_width=True,
            key=f"quick_{idx}",
            help=question
        ):
            st.session_state.last_question = question
            st.rerun()
        
        # Show the actual question text below the button in a subtle way
        st.caption(f"_{question}_")
        st.markdown("")  # Add spacing


def display_system_info():
    """Display system information in sidebar"""
    st.sidebar.markdown("## 🛠️ System Info")
    st.sidebar.info(
        f"""
        **Retrieval Settings:**
        - Top-k: {config.TOP_K} chunks
        - Threshold: {config.SIMILARITY_THRESHOLD}
        - Chunks loaded: {len(st.session_state.assistant.metadata)}
        """
    )
    
    st.sidebar.markdown("## 📊 Confidence Guide")
    st.sidebar.markdown(
        """
        - 🟢 **HIGH**: > 50% match
        - 🟡 **MEDIUM**: 25-50% match  
        - 🔴 **LOW**: < 25% match
        """
    )


def main():
    # Page configuration
    st.set_page_config(
        page_title="FYP Handbook Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced Custom CSS for better styling
    st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main app styling */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Main content area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        border: 2px solid #e2e8f0;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        color: #2d3748;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #2d3748;
    }
    
    /* Info box styling */
    .stAlert {
        background: linear-gradient(135deg, #ebf4ff 0%, #e0e7ff 100%);
        border-left: 4px solid #667eea;
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #2d3748;
        border: 1px solid #e2e8f0;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #ebf4ff 0%, #e0e7ff 100%);
        border-color: #667eea;
    }
    
    /* Caption text for questions */
    .stCaption {
        color: #718096;
        font-style: italic;
        font-size: 0.85rem;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .stChatInput:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-family: 'Courier New', monospace;
    }
    
    /* Column divider */
    [data-testid="column"] {
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">📚 FAST-NUCES FYP Handbook Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Get instant answers about FYP guidelines with proper page references</div>', unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Display system info in sidebar
    display_system_info()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.markdown("### 💬 Ask About FYP Guidelines")
        
        # Handle last_question from quick questions first
        if st.session_state.last_question:
            question = st.session_state.last_question
            st.session_state.last_question = ""
            
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Get answer
            with st.spinner("🔍 Searching handbook..."):
                answer, sources, max_score = st.session_state.assistant.answer_question(question)
            
            # Extract pages from sources
            pages = list(set(chunk['page'] for chunk in sources)) if sources else []
            
            # Add assistant response to chat
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer, 
                "pages": pages,
                "confidence": max_score
            })
            
            st.rerun()
        
        # Display chat history
        for chat in st.session_state.chat_history:
            display_chat_message(
                chat["role"], 
                chat["content"], 
                chat.get("pages", []),
                chat.get("confidence", 0.0)
            )
        
        # Display source details for the last assistant response
        if st.session_state.chat_history:
            last_response = [chat for chat in st.session_state.chat_history if chat["role"] == "assistant"]
            if last_response:
                last = last_response[-1]
                if last.get("confidence", 0) >= config.SIMILARITY_THRESHOLD:
                    # Get the question that led to this response
                    assistant_idx = len(st.session_state.chat_history) - 1 - st.session_state.chat_history[::-1].index(last)
                    if assistant_idx > 0:
                        user_question = st.session_state.chat_history[assistant_idx - 1]["content"]
                        _, sources, _ = st.session_state.assistant.answer_question(user_question)
                        if sources:
                            display_source_details(sources)
        
        # Question input
        question = st.chat_input("Type your question about FYP guidelines...")
        
        if question:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Get answer
            with st.spinner("🔍 Searching handbook..."):
                answer, sources, max_score = st.session_state.assistant.answer_question(question)
            
            # Extract pages from sources
            pages = list(set(chunk['page'] for chunk in sources)) if sources else []
            
            # Add assistant response to chat
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer, 
                "pages": pages,
                "confidence": max_score
            })
            
            st.rerun()
        
        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    with col2:
        # Quick questions and info
        display_quick_questions()
        
        # Statistics
        st.markdown("### 📈 Session Stats")
        total_questions = len([chat for chat in st.session_state.chat_history if chat["role"] == "user"])
        successful_answers = len([chat for chat in st.session_state.chat_history if chat["role"] == "assistant" and chat.get("confidence", 0) > 0])
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Questions Asked", total_questions)
        with col_stat2:
            st.metric("Answers Found", successful_answers)
        
        # Help section
        with st.expander("ℹ️ How to use this assistant"):
            st.markdown("""
            **Tips for best results:**
            - Ask specific questions about formatting, structure, or requirements
            - Use the quick questions for common topics
            - Check the source details to verify information
            - Look for page references to find the original content
            
            **Examples:**
            - "What are the margin requirements?"
            - "How should I format headings?"
            - "What chapters are required in an R&D report?"
            """)


if __name__ == "__main__":
    main()