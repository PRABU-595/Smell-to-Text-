"""
TF-IDF baseline model for smell-to-molecule prediction
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Dict, Tuple, Optional
import pickle
import json


class TFIDFBaseline:
    """TF-IDF + Logistic Regression baseline for smell-to-molecule."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple = (1, 3)):
        """
        Initialize TF-IDF baseline model.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: N-gram range for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
        self.classifier = MultiOutputClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs'
            )
        )
        self.mlb = MultiLabelBinarizer()
        self.is_fitted = False
        self.chemical_names = []
    
    def fit(self, descriptions: List[str], chemicals: List[List[str]]) -> 'TFIDFBaseline':
        """
        Fit the model on training data.
        
        Args:
            descriptions: List of smell descriptions
            chemicals: List of chemical lists (multi-label)
            
        Returns:
            Fitted model
        """
        # Vectorize descriptions
        X = self.vectorizer.fit_transform(descriptions)
        
        # Binarize labels
        y = self.mlb.fit_transform(chemicals)
        self.chemical_names = list(self.mlb.classes_)
        
        # Fit classifier
        self.classifier.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, descriptions: List[str]) -> np.ndarray:
        """
        Predict chemicals for descriptions.
        
        Args:
            descriptions: List of smell descriptions
            
        Returns:
            Binary prediction matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.vectorizer.transform(descriptions)
        return self.classifier.predict(X)
    
    def predict_proba(self, descriptions: List[str]) -> np.ndarray:
        """
        Predict probabilities for each chemical.
        
        Args:
            descriptions: List of smell descriptions
            
        Returns:
            Probability matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.vectorizer.transform(descriptions)
        
        # Get probabilities from each classifier
        probas = []
        for estimator in self.classifier.estimators_:
            try:
                proba = estimator.predict_proba(X)[:, 1]
            except:
                proba = estimator.predict(X)
            probas.append(proba)
        
        return np.array(probas).T
    
    def predict_top_k(self, description: str, k: int = 5) -> List[Dict]:
        """
        Predict top-k chemicals for a description.
        
        Args:
            description: Smell description
            k: Number of top predictions
            
        Returns:
            List of {chemical, probability} dicts
        """
        probas = self.predict_proba([description])[0]
        top_indices = np.argsort(probas)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'chemical': self.chemical_names[idx],
                'probability': float(probas[idx])
            })
        
        return results
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'mlb': self.mlb,
                'chemical_names': self.chemical_names
            }, f)
    
    def load(self, filepath: str) -> 'TFIDFBaseline':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.mlb = data['mlb']
        self.chemical_names = data['chemical_names']
        self.is_fitted = True
        
        return self
    
    def get_feature_importance(self, chemical: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important features for a chemical.
        
        Args:
            chemical: Chemical name
            top_n: Number of top features
            
        Returns:
            List of (feature, importance) tuples
        """
        if chemical not in self.chemical_names:
            return []
        
        idx = self.chemical_names.index(chemical)
        estimator = self.classifier.estimators_[idx]
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = estimator.coef_[0]
        
        top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]
        
        return [(feature_names[i], coefficients[i]) for i in top_indices]


if __name__ == '__main__':
    # Example usage
    descriptions = [
        "Fresh citrus with bergamot and lemon",
        "Warm woody sandalwood with cedar notes",
        "Sweet vanilla and caramel dessert fragrance",
        "Light floral jasmine and rose petals",
    ]
    
    chemicals = [
        ['Limonene', 'Citral', 'Linalool'],
        ['Santalol', 'Cedrene', 'Vetiverol'],
        ['Vanillin', 'Ethyl vanillin', 'Maltol'],
        ['Benzyl acetate', 'Geraniol', 'Linalool'],
    ]
    
    # Train model
    model = TFIDFBaseline()
    model.fit(descriptions, chemicals)
    
    # Test prediction
    test_desc = "A zesty lemon fragrance with woody undertones"
    predictions = model.predict_top_k(test_desc, k=3)
    
    print(f"Description: {test_desc}")
    print("Predictions:")
    for pred in predictions:
        print(f"  {pred['chemical']}: {pred['probability']:.2%}")
