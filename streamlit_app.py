import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Medical Chatbot", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #2E86AB; text-align: center; }
.chat-container { background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
.user-message { background-color: #2E86AB; color: white; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right; }
.bot-message { background-color: #A8DADC; color: black; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: left; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 Medical Chatbot MediScan</h1>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def init_rag():
    try:
        # Use SMALLER model - much faster download
        model_name = "microsoft/DialoGPT-small"  # 110MB instead of 351MB
        
        with st.spinner("Downloading medical AI model (110MB)... This may take 2-3 minutes"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
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
        
        # Check for FAISS index
        if os.path.exists("faiss_index"):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            prompt = PromptTemplate(template="Context: {context}\nQuestion: {question}\nAnswer:", 
                                   input_variables=["context", "question"])
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                chain_type_kwargs={"prompt": prompt}
            )
            return qa
        else:
            st.warning("FAISS index not found. Running in demo mode.")
            return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

qa_chain = init_rag()

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

user_input = st.text_input("Ask a medical question:", placeholder="e.g., I have fever and cough")

if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Analyzing..."):
        try:
            if qa_chain:
                response = qa_chain.invoke({"query": user_input})
                answer = response['result']
            else:
                answer = "Medical knowledge base not loaded. Please upload FAISS index."
        except Exception as e:
            answer = f"Error: {str(e)}"
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()