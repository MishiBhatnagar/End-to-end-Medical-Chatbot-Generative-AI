# medical_chatbot_app.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import sys
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ============= FIXED: ML Classifier Import =============
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
ML_CLASSIFIER_AVAILABLE = False

try:
    # Try to import from train_classifier first (updated version)
    from ml_classifier.train_classifier import SymptomDiseaseClassifier
    ML_CLASSIFIER_AVAILABLE = True
    print("✅ ML Classifier loaded from train_classifier")
except ImportError:
    try:
        # Fallback to symptom_classifier
        from ml_classifier.symptom_classifier import SymptomDiseaseClassifier
        ML_CLASSIFIER_AVAILABLE = True
        print("✅ ML Classifier loaded from symptom_classifier")
    except ImportError:
        print("⚠️ ML Classifier not available. Install with: pip install scikit-learn pandas")
        ML_CLASSIFIER_AVAILABLE = False
# ========================================================

# Set page config
st.set_page_config(
    page_title="Medical Chatbot MediScan",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "classifier" not in st.session_state:
    st.session_state.classifier = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        max-height: 500px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #2E86AB;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-message {
        background-color: #A8DADC;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .probability-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 5px 0;
    }
    .probability-fill {
        height: 20px;
        background-color: #2E86AB;
        border-radius: 10px;
        color: white;
        text-align: right;
        padding-right: 10px;
        line-height: 20px;
        font-size: 12px;
    }
    .high-confidence { color: #2ecc71; }
    .medium-confidence { color: #f39c12; }
    .low-confidence { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 Medical Chatbot MediScan</h1>', unsafe_allow_html=True)
st.markdown("Ask me any medical questions from the Medical Encyclopedia!")

# ============= FIXED: Sidebar with better classifier loading =============
with st.sidebar:
    st.header("🔧 System Status")
    
    # Check FAISS
    faiss_found = False
    for path in ["faiss_index", "../faiss_index", "Data/faiss_index"]:
        if os.path.exists(path):
            faiss_found = True
            st.success(f"✅ FAISS index ready ({path})")
            break
    if not faiss_found:
        st.error("❌ FAISS index missing")
    
    # Load ML Classifier with better error handling
    if ML_CLASSIFIER_AVAILABLE:
        if st.session_state.classifier is None:
            with st.spinner("Loading ML classifier..."):
                try:
                    # Try to load from saved model first
                    classifier = SymptomDiseaseClassifier()
                    if classifier.load_model("ml_classifier/models/symptom_classifier.pkl"):
                        st.session_state.classifier = classifier
                        st.success("✅ ML Classifier loaded (trained model)")
                    else:
                        # Fallback to rule-based
                        st.session_state.classifier = classifier
                        st.info("ℹ️ Using rule-based classifier (no trained model)")
                except Exception as e:
                    st.error(f"❌ ML Classifier failed: {str(e)[:50]}...")
        else:
            st.success("✅ ML Classifier ready")
    else:
        st.warning("⚠️ ML Classifier not installed")
        st.info("Run: pip install scikit-learn pandas")
    
    st.header("📋 Sample Questions")
    sample_questions = [
        "What is Acne?",
        "I have fever and cough",
        "Symptoms of diabetes",
        "I have headache and nausea",
        "Chest pain treatment"
    ]
    
    # Store clicked question in session state
    if "sample_clicked" not in st.session_state:
        st.session_state.sample_clicked = None
    
    for question in sample_questions:
        if st.button(question, key=f"btn_{question}"):
            st.session_state.sample_clicked = question
# ========================================================================

# Initialize QA system
@st.cache_resource(show_spinner=False)
def initialize_medical_bot():
    try:
        # Find FAISS index
        faiss_path = None
        for path in ["faiss_index", "../faiss_index", "Data/faiss_index"]:
            if os.path.exists(path):
                faiss_path = path
                break
        
        if not faiss_path:
            raise FileNotFoundError("FAISS index not found")
        
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load vector store
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        
        # Load LLM
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create QA chain
        prompt_template = """Use the following medical context to answer the question accurately.

Medical Context:
{context}

Question: {question}

Provide a clear, helpful medical answer based ONLY on the context above:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return None

# Initialize bot
if st.session_state.qa_chain is None:
    with st.spinner("Loading Medical Bot... This may take a minute."):
        st.session_state.qa_chain = initialize_medical_bot()

# ============= FIXED: Handle sample question click =============
if st.session_state.get('sample_clicked'):
    user_input = st.session_state.sample_clicked
    st.session_state.sample_clicked = None  # Reset
else:
    user_input = st.session_state.get('user_input', '')
# ===============================================================

# Main chat interface
if st.session_state.qa_chain:
    # Create two columns: Main chat (70%) and Analysis (30%)
    col_chat, col_analysis = st.columns([0.7, 0.3])
    
    with col_chat:
        # Display chat messages
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # User input
        col1, col2 = st.columns([4, 1])
        with col1:
            text_input = st.text_input(
                "Ask a medical question:", 
                key="user_input", 
                value=user_input,
                placeholder="e.g., I have fever and cough or What is diabetes?"
            )
        with col2:
            send_button = st.button("Send", use_container_width=True)
    
    with col_analysis:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🔬 Symptom Analysis")
        
        # ============= FIXED: Show ML predictions with better display =============
        if st.session_state.classifier:
            # Get the last user message for analysis
            last_user_msg = None
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                last_user_msg = st.session_state.messages[-1]["content"]
            elif text_input:
                last_user_msg = text_input
            
            if last_user_msg:
                # Get predictions
                detected_symptoms, predictions, all_probs = st.session_state.classifier.predict(last_user_msg)
                
                # Show detected symptoms
                if detected_symptoms:
                    st.write("**✅ Detected Symptoms:**")
                    symptom_cols = st.columns(2)
                    for i, symptom in enumerate(detected_symptoms[:6]):
                        col_idx = i % 2
                        with symptom_cols[col_idx]:
                            st.markdown(f"- {symptom.title()}")
                    
                    # Show predictions
                    if predictions:
                        st.write("**📊 Likely Conditions:**")
                        
                        # Create DataFrame for chart
                        chart_data = pd.DataFrame({
                            'Disease': [p['disease'] for p in predictions[:5]],
                            'Probability': [p['probability'] for p in predictions[:5]]
                        })
                        
                        # Show bar chart
                        st.bar_chart(chart_data.set_index('Disease'), height=200)
                        
                        # Show probability bars with confidence colors
                        for pred in predictions[:3]:
                            # Set color based on confidence
                            if pred['probability'] >= 70:
                                color = "#2ecc71"  # Green
                                confidence = "High"
                            elif pred['probability'] >= 40:
                                color = "#f39c12"  # Orange
                                confidence = "Medium"
                            else:
                                color = "#e74c3c"  # Red
                                confidence = "Low"
                            
                            st.markdown(f"**{pred['disease']}** <span style='color: {color};'>({confidence} Confidence)</span>", unsafe_allow_html=True)
                            prob = pred['probability']
                            bar_width = min(prob, 100)
                            st.markdown(f"""
                            <div style="background-color: #f0f0f0; border-radius: 10px; height: 25px; width: 100%; margin-bottom: 10px;">
                                <div style="background-color: {color}; width: {bar_width}%; height: 25px; border-radius: 10px; 
                                          color: white; text-align: right; padding-right: 10px; line-height: 25px;">
                                    {prob}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.caption("*Based on symptom patterns in medical textbook*")
                    else:
                        st.info("ℹ️ No confident predictions. Please provide more specific symptoms.")
                else:
                    st.info("ℹ️ No specific symptoms detected. Try describing your symptoms clearly.")
        else:
            st.info("⏳ ML Classifier loading...")
        # =====================================================================
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show disclaimer
        st.markdown("""
        <div style="background-color: #fff3cd; border-radius: 10px; padding: 15px; margin-top: 20px;">
            ⚠️ <strong>Medical Disclaimer:</strong><br>
            This is for educational purposes only. Always consult a healthcare professional for medical advice.
        </div>
        """, unsafe_allow_html=True)
    
    # ============= FIXED: Handle send button with better error handling =============
    if (send_button and text_input) or (user_input and not st.session_state.messages):
        query = text_input or user_input
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get bot response
        with st.spinner("🔍 Analyzing symptoms and searching medical knowledge..."):
            try:
                response = st.session_state.qa_chain.invoke({"query": query})
                bot_response = response['result']
                
                # Add ML insights to response if available
                if st.session_state.classifier:
                    symptoms, predictions, _ = st.session_state.classifier.predict(query)
                    if predictions and len(predictions) > 0:
                        top_pred = predictions[0]
                        confidence_icon = "🟢" if top_pred['probability'] >= 70 else "🟡" if top_pred['probability'] >= 40 else "🟠"
                        bot_response = f"{confidence_icon} **Most likely: {top_pred['disease']} ({top_pred['probability']}% confidence)**\n\n{bot_response}"
                
                # Add bot response
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(response['source_documents'][:2]):
                        st.write(f"**Source {i+1}:**")
                        st.write(f"Content: {doc.page_content[:200]}...")
                        st.write("---")
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()
    # =====================================================================
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

else:
    st.markdown("""
    <div class="error-box">
    <h3>❌ Medical bot failed to initialize</h3>
    <p>Please ensure:</p>
    <ol>
        <li>FAISS index exists (run trials.ipynb first)</li>
        <li>You have internet connection for model download</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("🏥 MediScan - AI-Powered Medical Information System | Based on Medical Encyclopedia")