# ml_classifier/symptom_classifier.py
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SymptomDiseaseClassifier:
    """
    Wrapper class for Streamlit app to load and use the classifier
    """
    def __init__(self, model_path="ml_classifier/models/symptom_classifier.pkl"):
        self.model = None
        self.symptoms_list = []
        self.diseases_list = []
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.symptoms_list = model_data['symptoms_list']
            self.diseases_list = model_data['diseases_list']
            return True
        except FileNotFoundError:
            print(f"⚠️ Model not found at {self.model_path}")
            return False
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return False
    
    def extract_symptoms(self, text):
        """Extract symptoms from user input"""
        text = text.lower()
        detected = []
        
        for symptom in self.symptoms_list:
            if symptom in text:
                detected.append(symptom)
            # Check for plural/singular variations
            elif symptom + "s" in text:
                detected.append(symptom)
            elif symptom.replace(" ", "_") in text:
                detected.append(symptom)
        
        return detected
    
    def create_vector(self, symptoms):
        """Create binary vector for symptoms"""
        vector = [0] * len(self.symptoms_list)
        symptom_to_idx = {s: i for i, s in enumerate(self.symptoms_list)}
        
        for symptom in symptoms:
            if symptom in symptom_to_idx:
                vector[symptom_to_idx[symptom]] = 1
        
        return [vector]
    
    def predict(self, user_input):
        """Predict disease probabilities"""
        if self.model is None:
            if not self.load_model():
                return None, [], {}
        
        # Extract symptoms
        detected_symptoms = self.extract_symptoms(user_input)
        
        if not detected_symptoms:
            return None, [], {}
        
        # Create feature vector
        X = self.create_vector(detected_symptoms)
        
        # Get probabilities
        try:
            probabilities = self.model.predict_proba(X)[0]
            
            # Get top 3 predictions
            top_indices = probabilities.argsort()[-3:][::-1]
            
            predictions = []
            for idx in top_indices:
                if probabilities[idx] > 0.1:
                    predictions.append({
                        'disease': self.diseases_list[idx].title(),
                        'probability': round(probabilities[idx] * 100, 1)
                    })
            
            # All probabilities for chart
            all_probs = {
                self.diseases_list[i].title(): round(probabilities[i] * 100, 1)
                for i in range(len(self.diseases_list))
                if probabilities[i] > 0.05
            }
            
            return detected_symptoms, predictions, all_probs
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return detected_symptoms, [], {}