# ml_classifier/utils/symptom_utils.py
import re
import json
import os
from typing import List, Dict, Tuple
import pickle

# Common medical symptoms (expanded list)
COMMON_SYMPTOMS = [
    "fever", "cough", "headache", "fatigue", "nausea", "vomiting", 
    "diarrhea", "chest pain", "shortness of breath", "sore throat", 
    "runny nose", "sneezing", "body aches", "muscle pain", "joint pain",
    "rash", "itching", "swelling", "dizziness", "fainting",
    "abdominal pain", "back pain", "neck pain", "seizure", "confusion",
    "blurred vision", "weight loss", "weight gain", "loss of appetite",
    "night sweats", "chills", "fatigue", "weakness", "numbness",
    "tingling", "palpitations", "anxiety", "depression", "insomnia"
]

def extract_symptom_disease_pairs(text_chunks: List[str]) -> List[Dict]:
    """
    Extract symptom-disease pairs from text chunks using pattern matching
    """
    training_examples = []
    
    # Pattern 1: "Symptoms of [Disease] include [Symptoms]"
    pattern1 = r"(?:symptoms|signs|manifestations)(?:\s+of|\s+for)?\s+([A-Za-z\s]+?)(?:\s+include|\s+are|\s+may include|\s+can include|\s+consist of)\s+(.+?)(?:\.|;|$)"
    
    # Pattern 2: "[Disease] is characterized by [Symptoms]"
    pattern2 = r"([A-Za-z\s]+?)(?:\s+is|\s+are)?\s+(?:characterized by|marked by|presents with|associated with)\s+(.+?)(?:\.|;|$)"
    
    # Pattern 3: "[Disease] causes [Symptoms]"
    pattern3 = r"([A-Za-z\s]+?)\s+(?:causes|leads to|results in)\s+(.+?)(?:\.|;|$)"
    
    for chunk in text_chunks:
        chunk_lower = chunk.lower()
        
        # Apply patterns
        for pattern in [pattern1, pattern2, pattern3]:
            matches = re.finditer(pattern, chunk_lower, re.IGNORECASE)
            for match in matches:
                disease = match.group(1).strip()
                symptom_text = match.group(2).strip()
                
                # Extract individual symptoms
                symptoms = []
                for symptom in COMMON_SYMPTOMS:
                    if symptom in symptom_text:
                        symptoms.append(symptom)
                
                # Only add if we found at least 2 symptoms
                if len(symptoms) >= 2:
                    training_examples.append({
                        "disease": disease,
                        "symptoms": symptoms,
                        "full_text": symptom_text,
                        "source_chunk": chunk[:200] + "..."
                    })
    
    return training_examples

def create_symptom_vector(symptoms: List[str], all_symptoms: List[str] = None) -> List[int]:
    """
    Convert list of symptoms to binary vector
    """
    if all_symptoms is None:
        all_symptoms = COMMON_SYMPTOMS
    
    vector = [0] * len(all_symptoms)
    symptom_to_idx = {symptom: i for i, symptom in enumerate(all_symptoms)}
    
    for symptom in symptoms:
        if symptom in symptom_to_idx:
            vector[symptom_to_idx[symptom]] = 1
    
    return vector

def extract_symptoms_from_user_input(user_input: str) -> List[str]:
    """
    Extract symptoms from user's natural language input
    """
    user_input = user_input.lower()
    detected_symptoms = []
    
    for symptom in COMMON_SYMPTOMS:
        if symptom in user_input:
            detected_symptoms.append(symptom)
        # Check for variations
        elif symptom.replace(" ", "_") in user_input:
            detected_symptoms.append(symptom)
        elif symptom.replace(" ", "-") in user_input:
            detected_symptoms.append(symptom)
    
    return detected_symptoms

def save_training_data(examples: List[Dict], filename: str = "ml_classifier/data/training_data.json"):
    """
    Save extracted training examples to JSON
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"✅ Saved {len(examples)} training examples to {filename}")

def load_training_data(filename: str = "ml_classifier/data/training_data.json") -> List[Dict]:
    """
    Load training examples from JSON
    """
    with open(filename, 'r') as f:
        return json.load(f)