# ml_classifier/train_classifier.py
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_classifier.utils.symptom_utils import (
    COMMON_SYMPTOMS, 
    create_symptom_vector,
    load_training_data,
    extract_symptoms_from_user_input
)

class SymptomDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.symptoms_list = COMMON_SYMPTOMS
        self.diseases_list = []
        self.rule_based_fallback = self._create_rule_based()
        
    def _create_rule_based(self):
        """Create rule-based disease-symptom mapping as fallback"""
        return {
            "influenza": ["fever", "cough", "fatigue", "body aches", "headache", "chills"],
            "common cold": ["runny nose", "sneezing", "sore throat", "cough", "congestion"],
            "covid-19": ["fever", "cough", "fatigue", "loss of taste", "loss of smell"],
            "pneumonia": ["fever", "cough", "shortness of breath", "chest pain"],
            "bronchitis": ["cough", "fatigue", "shortness of breath", "chest discomfort"],
            "asthma": ["shortness of breath", "cough", "chest tightness", "wheezing"],
            "migraine": ["headache", "nausea", "vomiting", "sensitivity to light"],
            "tension headache": ["headache", "neck pain", "shoulder tension", "scalp tenderness"],
            "gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal pain"],
            "food poisoning": ["nausea", "vomiting", "diarrhea", "stomach cramps"],
            "uti": ["burning urination", "frequent urination", "abdominal pain"],
            "kidney stones": ["severe back pain", "abdominal pain", "painful urination"],
            "allergies": ["sneezing", "runny nose", "itchy eyes", "cough"],
            "anaphylaxis": ["difficulty breathing", "swelling", "hives", "dizziness"],
            "diabetes": ["fatigue", "weight loss", "frequent urination", "excessive thirst"],
            "hypoglycemia": ["shaking", "sweating", "hunger", "dizziness"],
            "hypertension": ["headache", "dizziness", "blurred vision"],
            "angina": ["chest pain", "chest pressure", "shortness of breath"],
            "arthritis": ["joint pain", "joint swelling", "stiffness"],
            "fibromyalgia": ["widespread pain", "fatigue", "sleep problems"],
            "eczema": ["itchy skin", "red rash", "dry skin", "scaly patches"],
            "psoriasis": ["red patches", "silvery scales", "dry cracked skin"],
            "hives": ["raised welts", "itching", "swelling", "redness"],
            "anxiety": ["anxiety", "palpitations", "dizziness", "shortness of breath"],
            "depression": ["sadness", "loss of interest", "fatigue", "sleep changes"],
            "gerd": ["heartburn", "chest pain", "regurgitation", "acid reflux"],
            "pms": ["bloating", "breast tenderness", "mood swings", "fatigue"],
            "yeast infection": ["itching", "burning", "discharge", "redness"]
        }
        
    def prepare_data(self, training_examples):
        """
        Convert training examples to features and labels
        """
        X = []
        y = []
        
        # Get unique diseases
        diseases = {}
        for ex in training_examples:
            disease = ex['disease'].lower().strip()
            if disease not in diseases:
                diseases[disease] = len(diseases)
        
        self.diseases_list = list(diseases.keys())
        print(f"📊 Found {len(self.diseases_list)} unique diseases")
        
        # Create feature vectors
        for ex in training_examples:
            # Create symptom vector
            symptom_vector = create_symptom_vector(ex['symptoms'], self.symptoms_list)
            X.append(symptom_vector)
            
            # Get disease label
            disease = ex['disease'].lower().strip()
            y.append(diseases[disease])
        
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """
        Train Random Forest classifier with small dataset handling
        """
        print("\n🌲 Training Random Forest Classifier...")
        
        # Check if we have enough data for train/test split
        unique_classes = len(np.unique(y))
        samples_per_class = np.bincount(y)
        min_samples_per_class = np.min(samples_per_class)
        
        # If dataset is too small, train on all data
        if len(X) < 20 or min_samples_per_class < 2:
            print(f"⚠️ Small dataset detected ({len(X)} samples). Training on all data without validation split.")
            
            self.model = RandomForestClassifier(
                n_estimators=50,  # Fewer trees for small dataset
                max_depth=5,      # Shallower trees to prevent overfitting
                random_state=42,
                class_weight='balanced',
                min_samples_split=2,
                min_samples_leaf=1
            )
            self.model.fit(X, y)
            
            # Self-test on training data (just for info)
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            print(f"✅ Model trained on all {len(X)} samples")
            print(f"📈 Training accuracy: {accuracy:.2%}")
            print(f"⚠️ Note: This is training accuracy, not validation accuracy")
            
        else:
            # Normal train/test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced',
                    min_samples_split=2,
                    min_samples_leaf=1
                )
                self.model.fit(X_train, y_train)
                
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"✅ Model trained!")
                print(f"📈 Test accuracy: {accuracy:.2%}")
                print(f"📊 Training samples: {len(X_train)}")
                print(f"🧪 Test samples: {len(X_test)}")
                
            except Exception as e:
                print(f"⚠️ Error during train/test split: {e}")
                print("Falling back to training on all data...")
                
                self.model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42,
                    class_weight='balanced'
                )
                self.model.fit(X, y)
                accuracy = 0.0
        
        # Add rule-based diseases to diseases_list if not present
        for disease in self.rule_based_fallback.keys():
            if disease.title() not in self.diseases_list:
                self.diseases_list.append(disease.title())
        
        return accuracy
    
    def predict(self, symptoms_text):
        """
        Predict disease from symptom description
        Uses ML model if available, falls back to rule-based if not
        """
        # Extract symptoms from text
        detected_symptoms = extract_symptoms_from_user_input(symptoms_text)
        
        if not detected_symptoms:
            return None, [], {}
        
        # Try ML prediction first
        if self.model is not None and len(self.diseases_list) > 0:
            try:
                # Create feature vector
                X = create_symptom_vector(detected_symptoms, self.symptoms_list)
                X = np.array([X])
                
                # Get probabilities
                probabilities = self.model.predict_proba(X)[0]
                
                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[-3:][::-1]
                
                predictions = []
                for idx in top_indices:
                    if probabilities[idx] > 0.1:  # Only show if >10% confidence
                        disease_name = self.diseases_list[idx] if idx < len(self.diseases_list) else "Unknown"
                        predictions.append({
                            'disease': disease_name.title() if disease_name != "Unknown" else disease_name,
                            'probability': round(probabilities[idx] * 100, 1),
                            'confidence': 'High' if probabilities[idx] > 0.7 else 'Medium' if probabilities[idx] > 0.4 else 'Low'
                        })
                
                # All probabilities for visualization
                all_probs = {}
                for i in range(min(len(self.diseases_list), len(probabilities))):
                    if probabilities[i] > 0.05:
                        all_probs[self.diseases_list[i].title()] = round(probabilities[i] * 100, 1)
                
                # If ML predictions are weak, fall back to rule-based
                if not predictions or predictions[0]['probability'] < 30:
                    return self._rule_based_predict(detected_symptoms)
                
                return detected_symptoms, predictions, all_probs
                
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}. Using rule-based fallback.")
                return self._rule_based_predict(detected_symptoms)
        else:
            # No ML model, use rule-based
            return self._rule_based_predict(detected_symptoms)
    
    def _rule_based_predict(self, detected_symptoms):
        """
        Rule-based prediction fallback
        """
        # Calculate match scores for each disease
        scores = {}
        for disease, symptoms in self.rule_based_fallback.items():
            matches = sum(1 for s in detected_symptoms if s in symptoms)
            if matches > 0:
                # Calculate percentage based on how many disease symptoms matched
                score = (matches / len(symptoms)) * 100
                scores[disease.title()] = round(score, 1)
        
        # Sort by score
        sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        
        # Get top 3 predictions
        predictions = []
        for disease, score in list(sorted_scores.items())[:3]:
            predictions.append({
                'disease': disease,
                'probability': score,
                'confidence': 'High' if score > 70 else 'Medium' if score > 40 else 'Low'
            })
        
        # If no matches, return empty
        if not predictions:
            return detected_symptoms, [], {}
        
        return detected_symptoms, predictions, sorted_scores
    
    def save_model(self, path="ml_classifier/models/symptom_classifier.pkl"):
        """
        Save trained model to disk
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'symptoms_list': self.symptoms_list,
            'diseases_list': self.diseases_list,
            'rule_based_fallback': self.rule_based_fallback
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path="ml_classifier/models/symptom_classifier.pkl"):
        """
        Load trained model from disk
        """
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model', None)
            self.symptoms_list = model_data.get('symptoms_list', COMMON_SYMPTOMS)
            self.diseases_list = model_data.get('diseases_list', [])
            self.rule_based_fallback = model_data.get('rule_based_fallback', self._create_rule_based())
            
            print(f"✅ Model loaded from {path}")
            return self
        except FileNotFoundError:
            print(f"⚠️ Model not found at {path}. Using rule-based only.")
            return self
        except Exception as e:
            print(f"⚠️ Error loading model: {e}. Using rule-based only.")
            return self

def main():
    print("="*60)
    print("🤖 SYMPTOM DISEASE CLASSIFIER TRAINER")
    print("="*60)
    
    # Initialize classifier
    classifier = SymptomDiseaseClassifier()
    
    # Try to load training data
    try:
        training_examples = load_training_data()
        print(f"✅ Loaded {len(training_examples)} training examples")
        
        if len(training_examples) > 5:
            # Prepare data
            X, y = classifier.prepare_data(training_examples)
            
            # Train model
            accuracy = classifier.train(X, y)
            
            # Save model
            classifier.save_model()
        else:
            print(f"⚠️ Only {len(training_examples)} training examples. Not enough for ML training.")
            print("📝 Saving rule-based classifier only...")
            classifier.save_model()
            
    except FileNotFoundError:
        print("❌ No training data found. Creating rule-based classifier...")
        classifier.save_model()
    
    # Test with some examples
    print("\n🧪 TESTING EXAMPLES:")
    test_queries = [
        "I have fever and cough",
        "I have runny nose and sneezing",
        "I have headache and nausea",
        "I have chest pain and shortness of breath",
        "I have fatigue and joint pain"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        symptoms, predictions, all_probs = classifier.predict(query)
        print(f"   Detected symptoms: {symptoms}")
        
        if predictions:
            print(f"   Predictions:")
            for pred in predictions[:3]:
                bar = "█" * int(min(pred['probability'] / 5, 20))
                confidence_color = "🟢" if pred['confidence'] == 'High' else "🟡" if pred['confidence'] == 'Medium' else "🟠"
                print(f"     - {confidence_color} {pred['disease']}: {pred['probability']}% {bar}")
        else:
            print("   No confident predictions. Please provide more specific symptoms.")
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("📌 Your classifier is ready with rule-based fallback!")
    print("📁 Model saved to: ml_classifier/models/symptom_classifier.pkl")
    print("="*60)

if __name__ == "__main__":
    main()