# ml_classifier/extract_training_data.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ml_classifier.utils.symptom_utils import (
    extract_symptom_disease_pairs, 
    save_training_data,
    COMMON_SYMPTOMS
)
import json

def load_text_chunks_from_faiss():
    """
    Load all text chunks from your existing FAISS index
    """
    print("📚 Loading FAISS index...")
    
    # Check multiple possible locations
    possible_paths = [
        "faiss_index",
        "../faiss_index",
        "./faiss_index",
        "Data/faiss_index"
    ]
    
    faiss_path = None
    for path in possible_paths:
        if os.path.exists(path):
            faiss_path = path
            print(f"✅ Found FAISS index at: {path}")
            break
    
    if not faiss_path:
        print("❌ FAISS index not found!")
        return []
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        
        # Extract text chunks from the vectorstore
        text_chunks = []
        
        # Try to get documents from the index
        if hasattr(vectorstore, 'docstore'):
            for doc_id in vectorstore.index_to_docstore_id.values():
                if doc_id in vectorstore.docstore._dict:
                    doc = vectorstore.docstore._dict[doc_id]
                    text_chunks.append(doc.page_content)
        
        print(f"✅ Loaded {len(text_chunks)} text chunks")
        return text_chunks
        
    except Exception as e:
        print(f"❌ Error loading FAISS: {e}")
        return []

# In ml_classifier/extract_training_data.py, replace the manual_training_examples function:

def manual_training_examples():
    """
    Provide expanded manual training examples
    """
    print("📝 Adding expanded manual training examples...")
    
    manual_examples = [
        # Respiratory diseases
        {"disease": "influenza", "symptoms": ["fever", "cough", "fatigue", "body aches", "headache", "chills"]},
        {"disease": "influenza", "symptoms": ["fever", "sore throat", "cough", "muscle pain", "fatigue"]},
        {"disease": "influenza", "symptoms": ["high fever", "dry cough", "headache", "weakness"]},
        
        {"disease": "common cold", "symptoms": ["runny nose", "sneezing", "sore throat", "cough", "congestion"]},
        {"disease": "common cold", "symptoms": ["nasal congestion", "sneezing", "watery eyes", "mild cough"]},
        {"disease": "common cold", "symptoms": ["scratchy throat", "runny nose", "sneezing", "mild fatigue"]},
        
        {"disease": "covid-19", "symptoms": ["fever", "cough", "fatigue", "loss of taste", "loss of smell"]},
        {"disease": "covid-19", "symptoms": ["shortness of breath", "cough", "fever", "body aches"]},
        {"disease": "covid-19", "symptoms": ["sore throat", "congestion", "fever", "headache", "fatigue"]},
        
        {"disease": "pneumonia", "symptoms": ["fever", "cough", "shortness of breath", "chest pain", "fatigue"]},
        {"disease": "pneumonia", "symptoms": ["productive cough", "fever", "chills", "difficulty breathing"]},
        {"disease": "pneumonia", "symptoms": ["chest pain when breathing", "cough with phlegm", "fever", "sweating"]},
        
        {"disease": "bronchitis", "symptoms": ["cough", "fatigue", "shortness of breath", "chest discomfort"]},
        {"disease": "bronchitis", "symptoms": ["persistent cough", "mucus production", "wheezing", "mild fever"]},
        {"disease": "bronchitis", "symptoms": ["dry cough", "chest tightness", "sore throat", "body aches"]},
        
        {"disease": "asthma", "symptoms": ["shortness of breath", "cough", "chest tightness", "wheezing"]},
        {"disease": "asthma", "symptoms": ["difficulty breathing", "wheezing", "cough at night", "chest pressure"]},
        {"disease": "asthma", "symptoms": ["coughing fits", "tight chest", "shortness of breath", "wheezing"]},
        
        # Neurological
        {"disease": "migraine", "symptoms": ["headache", "nausea", "vomiting", "sensitivity to light", "sensitivity to sound"]},
        {"disease": "migraine", "symptoms": ["throbbing headache", "aura", "blurred vision", "nausea"]},
        {"disease": "migraine", "symptoms": ["severe headache", "dizziness", "sensitivity to light", "vomiting"]},
        
        {"disease": "tension headache", "symptoms": ["headache", "neck pain", "shoulder tension", "scalp tenderness"]},
        {"disease": "tension headache", "symptoms": ["dull headache", "pressure forehead", "tightness around head"]},
        {"disease": "tension headache", "symptoms": ["mild headache", "stress", "fatigue", "muscle tension"]},
        
        # Gastrointestinal
        {"disease": "gastroenteritis", "symptoms": ["nausea", "vomiting", "diarrhea", "abdominal pain", "fever"]},
        {"disease": "gastroenteritis", "symptoms": ["stomach cramps", "watery diarrhea", "nausea", "dehydration"]},
        {"disease": "gastroenteritis", "symptoms": ["vomiting", "diarrhea", "abdominal pain", "loss of appetite"]},
        
        {"disease": "food poisoning", "symptoms": ["nausea", "vomiting", "diarrhea", "stomach cramps", "fever"]},
        {"disease": "food poisoning", "symptoms": ["sudden nausea", "vomiting", "watery diarrhea", "abdominal pain"]},
        {"disease": "food poisoning", "symptoms": ["vomiting", "diarrhea", "fever", "chills", "body aches"]},
        
        {"disease": "gerd", "symptoms": ["heartburn", "chest pain", "regurgitation", "difficulty swallowing"]},
        {"disease": "gerd", "symptoms": ["acid reflux", "burning in chest", "sour taste", "cough"]},
        {"disease": "gerd", "symptoms": ["chest discomfort", "regurgitation", "nausea", "sore throat"]},
        
        # Cardiovascular
        {"disease": "hypertension", "symptoms": ["headache", "dizziness", "blurred vision", "nosebleeds"]},
        {"disease": "hypertension", "symptoms": ["chest pain", "shortness of breath", "headache", "fatigue"]},
        {"disease": "hypertension", "symptoms": ["severe headache", "anxiety", "shortness of breath", "flushing"]},
        
        {"disease": "angina", "symptoms": ["chest pain", "chest pressure", "shortness of breath", "arm pain"]},
        {"disease": "angina", "symptoms": ["discomfort in chest", "jaw pain", "back pain", "nausea"]},
        {"disease": "angina", "symptoms": ["chest tightness", "shoulder pain", "dizziness", "sweating"]},
        
        # Urinary
        {"disease": "uti", "symptoms": ["burning urination", "frequent urination", "abdominal pain", "cloudy urine"]},
        {"disease": "uti", "symptoms": ["urgent urination", "pelvic pain", "blood in urine", "fever"]},
        {"disease": "uti", "symptoms": ["painful urination", "lower back pain", "frequent urges", "foul urine"]},
        
        {"disease": "kidney stones", "symptoms": ["severe back pain", "abdominal pain", "painful urination", "blood in urine"]},
        {"disease": "kidney stones", "symptoms": ["flank pain", "nausea", "vomiting", "frequent urination"]},
        {"disease": "kidney stones", "symptoms": ["sharp pain", "groin pain", "burning urination", "fever"]},
        
        # Metabolic
        {"disease": "diabetes", "symptoms": ["fatigue", "weight loss", "frequent urination", "excessive thirst", "blurred vision"]},
        {"disease": "diabetes", "symptoms": ["increased hunger", "slow healing", "tingling hands", "dry mouth"]},
        {"disease": "diabetes", "symptoms": ["frequent infections", "blurred vision", "fatigue", "thirst"]},
        
        {"disease": "hypoglycemia", "symptoms": ["shaking", "sweating", "hunger", "anxiety", "confusion"]},
        {"disease": "hypoglycemia", "symptoms": ["dizziness", "palpitations", "headache", "weakness"]},
        {"disease": "hypoglycemia", "symptoms": ["irritability", "blurred vision", "fatigue", "nausea"]},
        
        # Musculoskeletal
        {"disease": "arthritis", "symptoms": ["joint pain", "joint swelling", "stiffness", "reduced range of motion"]},
        {"disease": "arthritis", "symptoms": ["morning stiffness", "joint tenderness", "warm joints", "fatigue"]},
        {"disease": "arthritis", "symptoms": ["pain in hands", "pain in knees", "swelling", "difficulty moving"]},
        
        {"disease": "fibromyalgia", "symptoms": ["widespread pain", "fatigue", "sleep problems", "brain fog"]},
        {"disease": "fibromyalgia", "symptoms": ["muscle pain", "tender points", "anxiety", "headaches"]},
        {"disease": "fibromyalgia", "symptoms": ["chronic pain", "fatigue", "memory issues", "mood changes"]},
        
        # Allergic
        {"disease": "allergies", "symptoms": ["sneezing", "runny nose", "itchy eyes", "cough", "congestion"]},
        {"disease": "allergies", "symptoms": ["watery eyes", "nasal congestion", "sneezing", "itchy throat"]},
        {"disease": "allergies", "symptoms": ["post-nasal drip", "cough", "fatigue", "headache"]},
        
        {"disease": "anaphylaxis", "symptoms": ["difficulty breathing", "swelling", "hives", "dizziness", "rapid pulse"]},
        {"disease": "anaphylaxis", "symptoms": ["wheezing", "tight throat", "rash", "low blood pressure", "fainting"]},
        {"disease": "anaphylaxis", "symptoms": ["shortness of breath", "tongue swelling", "vomiting", "confusion"]},
        
        # Skin conditions
        {"disease": "eczema", "symptoms": ["itchy skin", "red rash", "dry skin", "scaly patches", "cracked skin"]},
        {"disease": "eczema", "symptoms": ["inflamed skin", "itching", "blisters", "leathery patches"]},
        {"disease": "eczema", "symptoms": ["dry sensitive skin", "intense itching", "dark colored patches"]},
        
        {"disease": "psoriasis", "symptoms": ["red patches", "silvery scales", "dry cracked skin", "itching", "burning"]},
        {"disease": "psoriasis", "symptoms": ["thickened nails", "scaly scalp", "joint pain", "swollen fingers"]},
        {"disease": "psoriasis", "symptoms": ["plaque buildup", "skin flaking", "discomfort", "bleeding"]},
        
        {"disease": "hives", "symptoms": ["raised welts", "itching", "swelling", "redness", "warmth"]},
        {"disease": "hives", "symptoms": ["itchy bumps", "angioedema", "burning", "wheals"]},
        {"disease": "hives", "symptoms": ["red swollen skin", "itching", "welts change shape", "discomfort"]},
        
        # Mental health
        {"disease": "anxiety", "symptoms": ["anxiety", "palpitations", "dizziness", "shortness of breath", "sweating"]},
        {"disease": "anxiety", "symptoms": ["worry", "restlessness", "fatigue", "muscle tension", "insomnia"]},
        {"disease": "anxiety", "symptoms": ["panic attacks", "fear", "nausea", "numbness", "chest tightness"]},
        
        {"disease": "depression", "symptoms": ["sadness", "loss of interest", "fatigue", "sleep changes", "appetite changes"]},
        {"disease": "depression", "symptoms": ["hopelessness", "low energy", "poor concentration", "weight changes"]},
        {"disease": "depression", "symptoms": ["irritability", "withdrawal", "aches", "feelings of worthlessness"]},
        
        # Women's health
        {"disease": "pms", "symptoms": ["bloating", "breast tenderness", "mood swings", "fatigue", "headache"]},
        {"disease": "pms", "symptoms": ["irritability", "anxiety", "cramping", "food cravings", "acne"]},
        {"disease": "pms", "symptoms": ["emotional changes", "insomnia", "joint pain", "constipation"]},
        
        {"disease": "yeast infection", "symptoms": ["itching", "burning", "discharge", "redness", "pain during sex"]},
        {"disease": "yeast infection", "symptoms": ["thick discharge", "vulvar swelling", "soreness", "rash"]},
        {"disease": "yeast infection", "symptoms": ["irritation", "burning during urination", "white discharge"]},
    ]
    
    # Format to match our structure
    formatted_examples = []
    for ex in manual_examples:
        formatted_examples.append({
            "disease": ex["disease"],
            "symptoms": ex["symptoms"],
            "full_text": ", ".join(ex["symptoms"]),
            "source_chunk": "Manual training example"
        })
    
    print(f"✅ Added {len(formatted_examples)} manual training examples")
    return formatted_examples
def main():
    print("="*60)
    print("🔍 MEDICAL TRAINING DATA EXTRACTOR")
    print("="*60)
    
    # First try to load from FAISS
    text_chunks = load_text_chunks_from_faiss()
    
    if text_chunks:
        # Extract from PDF
        print("\n🔬 Extracting symptom-disease pairs from textbook...")
        training_examples = extract_symptom_disease_pairs(text_chunks)
        print(f"✅ Extracted {len(training_examples)} examples from textbook")
    else:
        print("\n⚠️ Could not load from FAISS, using manual examples...")
        training_examples = manual_training_examples()
    
    # Add manual examples for better coverage
    manual_examples = manual_training_examples()
    all_examples = training_examples + manual_examples
    
    # Remove duplicates based on disease name
    unique_examples = {}
    for ex in all_examples:
        disease = ex["disease"].lower()
        if disease not in unique_examples:
            unique_examples[disease] = ex
    
    final_examples = list(unique_examples.values())
    print(f"\n📊 Total unique training examples: {len(final_examples)}")
    
    # Save to file
    save_training_data(final_examples)
    
    # Print summary
    print("\n📋 DISEASES COVERED:")
    for i, ex in enumerate(final_examples[:10], 1):
        print(f"  {i}. {ex['disease'].title()}: {', '.join(ex['symptoms'][:3])}...")
    
    if len(final_examples) > 10:
        print(f"  ... and {len(final_examples)-10} more")
    
    print("\n✅ Extraction complete!")
    print("   Next step: Run 'python ml_classifier/train_classifier.py'")

if __name__ == "__main__":
    main()