"""
Model Training Module for Customer Churn Prediction
Trains multiple models and selects the best performer
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix, roc_curve)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Train and evaluate multiple churn prediction models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self, data_path='data/processed/'):
        """Load processed training data"""
        logger.info(f"Loading data from {data_path}")
        
        X_train = pd.read_csv(f'{data_path}/X_train.csv')
        X_test = pd.read_csv(f'{data_path}/X_test.csv')
        y_train = pd.read_csv(f'{data_path}/y_train.csv').values.ravel()
        y_test = pd.read_csv(f'{data_path}/y_test.csv').values.ravel()
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """
        Handle class imbalance
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'smote', 'none', or 'class_weight'
            
        Returns:
            Resampled X_train, y_train
        """
        if method == 'smote':
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Class distribution: {np.bincount(y_train)}")
        
        return X_train, y_train
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression"""
        logger.info("Training Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        self.models['Logistic Regression'] = model
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        self.models['Random Forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost"""
        logger.info("Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        self.models['XGBoost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM"""
        logger.info("Training LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        self.models['LightGBM'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting"""
        logger.info("Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        self.models['Gradient Boosting'] = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = metrics
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test, use_smote=True):
        """
        Train all models and evaluate them
        
        Args:
            X_train, y_train, X_test, y_test: Train/test data
            use_smote: Whether to use SMOTE for imbalance
        """
        # Handle imbalance
        if use_smote:
            X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        # Train models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Evaluate all models
        logger.info("\n" + "="*50)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*50)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        # Find best model
        self.select_best_model()
    
    def select_best_model(self):
        """Select the best model based on F1 score"""
        best_f1 = 0
        
        for model_name, metrics in self.results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"BEST MODEL: {self.best_model_name}")
        logger.info(f"F1 Score: {best_f1:.4f}")
        logger.info(f"{'='*50}\n")
    
    def plot_model_comparison(self, save_path='reports/'):
        """Create comparison plots for all models"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame(self.results).T
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            metrics_df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}/model_comparison.png")
        plt.close()
        
        # Create ROC-AUC comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        roc_scores = metrics_df['roc_auc'].sort_values(ascending=False)
        roc_scores.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
        ax.set_title('ROC-AUC Score Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('ROC-AUC Score')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/roc_auc_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"ROC-AUC comparison plot saved to {save_path}/roc_auc_comparison.png")
        plt.close()
    
    def plot_confusion_matrix(self, X_test, y_test, save_path='reports/'):
        """Plot confusion matrix for best model"""
        if self.best_model is None:
            logger.warning("No best model selected yet!")
            return
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Not Churn', 'Churn'],
                   yticklabels=['Not Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}/confusion_matrix.png")
        plt.close()
    
    def save_best_model(self, save_path='data/models/'):
        """Save the best model"""
        if self.best_model is None:
            logger.warning("No best model to save!")
            return
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        model_path = f'{save_path}/best_model.pkl'
        joblib.dump(self.best_model, model_path)
        
        # Save model info
        info = {
            'model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name]
        }
        joblib.dump(info, f'{save_path}/model_info.pkl')
        
        logger.info(f"Best model ({self.best_model_name}) saved to {model_path}")
    
    def generate_classification_report(self, X_test, y_test, save_path='reports/'):
        """Generate detailed classification report"""
        if self.best_model is None:
            logger.warning("No best model selected yet!")
            return
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        y_pred = self.best_model.predict(X_test)
        
        report = classification_report(y_test, y_pred, 
                                      target_names=['Not Churn', 'Churn'],
                                      output_dict=True)
        
        # Save as DataFrame
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'{save_path}/classification_report.csv')
        
        logger.info(f"Classification report saved to {save_path}/classification_report.csv")
        
        # Print report
        print("\n" + "="*60)
        print(f"CLASSIFICATION REPORT - {self.best_model_name}")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))


def main():
    """Main training pipeline"""
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    # Train all models
    trainer.train_all_models(X_train, y_train, X_test, y_test, use_smote=True)
    
    # Generate visualizations
    trainer.plot_model_comparison()
    trainer.plot_confusion_matrix(X_test, y_test)
    
    # Save best model
    trainer.save_best_model()
    
    # Generate classification report
    trainer.generate_classification_report(X_test, y_test)
    
    print("\n✓ Model training complete!")
    print(f"✓ Best model: {trainer.best_model_name}")
    print(f"✓ Model saved to data/models/")
    print(f"✓ Reports saved to reports/")


if __name__ == "__main__":
    main()
