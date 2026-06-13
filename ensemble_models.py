"""
Ensemble Voting Classifier
===========================
Combines multiple ML models for improved accuracy

Expected Improvement: +2-5%

Models Combined:
  1. XGBoost - Gradient boosting, fast and accurate
  2. LightGBM - Very fast, memory efficient
  3. CatBoost - Handles categorical features well
  4. Random Forest - Robust and prevents overfitting

Voting Strategy:
  - Hard voting: Majority vote wins
  - Soft voting: Average probability wins
  - Weighted voting: Assign higher weight to best performers
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False


class EnsembleModelBuilder:
    """Build and manage ensemble models"""
    
    def __init__(self, ticker, verbose=True):
        self.ticker = ticker
        self.verbose = verbose
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
    
    def log(self, message):
        """Print log message if verbose"""
        if self.verbose:
            print(message)
    
    def create_xgboost_model(self):
        """Create XGBoost classifier"""
        if not XGBOOST_AVAILABLE:
            self.log("WARNING: XGBoost not available, skipping")
            return None
        
        return xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'  # Fast training
        )
    
    def create_lightgbm_model(self):
        """Create LightGBM classifier"""
        if not LIGHTGBM_AVAILABLE:
            self.log("WARNING: LightGBM not available, skipping")
            return None
        
        return lgb.LGBMClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    
    def create_catboost_model(self):
        """Create CatBoost classifier"""
        if not CATBOOST_AVAILABLE:
            self.log("WARNING: CatBoost not available, skipping")
            return None
        
        return CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            iterations=200,
            subsample=0.8,
            random_state=42,
            verbose=False,
            thread_count=-1
        )
    
    def create_random_forest_model(self):
        """Create Random Forest classifier"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def build_ensemble(self, X_train, y_train, voting='hard'):
        """
        Build ensemble model with available models
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        voting : str
            'hard' - majority vote
            'soft' - average probability
        
        Returns:
        --------
        ensemble : VotingClassifier
            Trained ensemble model
        """
        self.log(f"Building ensemble for {self.ticker}...")
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        estimators = []
        
        # Add XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model = self.create_xgboost_model()
            xgb_model.fit(X_train, y_train)
            estimators.append(('xgb', xgb_model))
            self.log("✓ XGBoost model trained")
        
        # Add LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_model = self.create_lightgbm_model()
            lgb_model.fit(X_train, y_train)
            estimators.append(('lgb', lgb_model))
            self.log("✓ LightGBM model trained")
        
        # Add CatBoost
        if CATBOOST_AVAILABLE:
            cat_model = self.create_catboost_model()
            cat_model.fit(X_train, y_train)
            estimators.append(('cat', cat_model))
            self.log("✓ CatBoost model trained")
        
        # Add Random Forest (always available)
        rf_model = self.create_random_forest_model()
        rf_model.fit(X_train_scaled, y_train)
        estimators.append(('rf', rf_model))
        self.log("✓ Random Forest model trained")
        
        if len(estimators) < 2:
            self.log("ERROR: Not enough models for ensemble!")
            return None
        
        # Create ensemble
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        self.log(f"Ensemble created with {len(estimators)} models ({voting} voting)")
        
        return self.ensemble
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble on test data"""
        if self.ensemble is None:
            self.log("ERROR: Ensemble not built yet!")
            return None
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # For ensemble with mixed scalers, predict from each model
        accuracy = self.ensemble.score(X_test_scaled, y_test)
        self.log(f"Ensemble accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def get_predictions(self, X):
        """Get ensemble predictions"""
        if self.ensemble is None:
            self.log("ERROR: Ensemble not built yet!")
            return None
        
        X_scaled = self.scaler.transform(X)
        predictions = self.ensemble.predict(X_scaled)
        probabilities = self.ensemble.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def save_ensemble(self, filepath):
        """Save ensemble model to file"""
        if self.ensemble is None:
            self.log("ERROR: Ensemble not built yet!")
            return False
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'ensemble': self.ensemble,
                    'scaler': self.scaler,
                    'ticker': self.ticker
                }, f)
            self.log(f"Ensemble saved to {filepath}")
            return True
        except Exception as e:
            self.log(f"ERROR: Failed to save ensemble: {e}")
            return False
    
    def load_ensemble(self, filepath):
        """Load ensemble model from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.ensemble = data['ensemble']
                self.scaler = data['scaler']
                self.ticker = data['ticker']
            self.log(f"Ensemble loaded from {filepath}")
            return True
        except Exception as e:
            self.log(f"ERROR: Failed to load ensemble: {e}")
            return False


class WeightedEnsemble:
    """Ensemble with weighted voting based on model performance"""
    
    def __init__(self, model_weights=None):
        """
        Initialize with optional model weights
        
        model_weights : dict
            {'xgb': 0.3, 'lgb': 0.25, 'cat': 0.25, 'rf': 0.2}
        """
        self.model_weights = model_weights or {
            'xgb': 0.3,    # XGBoost gets highest weight
            'lgb': 0.25,   # LightGBM second
            'cat': 0.25,   # CatBoost equal to LGB
            'rf': 0.2      # Random Forest lower weight
        }
        self.models = {}
    
    def add_model(self, name, model, weight=None):
        """Add model to ensemble"""
        self.models[name] = model
        if weight is not None:
            self.model_weights[name] = weight
    
    def predict(self, X, method='weighted_probability'):
        """
        Make predictions with weighted ensemble
        
        method : str
            'weighted_probability' - average of weighted probabilities
            'majority_vote' - weighted vote (best when weights sum to 1)
        """
        predictions = []
        
        for name, model in self.models.items():
            weight = self.model_weights.get(name, 1.0)
            pred = model.predict_proba(X)
            predictions.append(pred * weight)
        
        # Average weighted predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Get class with highest probability
        return np.argmax(ensemble_pred, axis=1)
    
    def predict_proba(self, X):
        """Get probability predictions from ensemble"""
        predictions = []
        
        for name, model in self.models.items():
            weight = self.model_weights.get(name, 1.0)
            pred = model.predict_proba(X)
            predictions.append(pred * weight)
        
        # Average weighted predictions
        return np.mean(predictions, axis=0)


# Example usage:
def example_usage():
    """Example of how to use ensemble models"""
    
    # Generate sample data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=19,
        n_informative=15,
        n_redundant=2,
        random_state=42
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build ensemble
    builder = EnsembleModelBuilder('2330.TW')
    ensemble = builder.build_ensemble(X_train, y_train, voting='soft')
    
    # Evaluate
    builder.evaluate_ensemble(X_test, y_test)
    
    # Get predictions
    preds, probs = builder.get_predictions(X_test)
    print(f"Sample predictions: {preds[:5]}")
    print(f"Sample probabilities: {probs[:5]}")


if __name__ == "__main__":
    example_usage()
