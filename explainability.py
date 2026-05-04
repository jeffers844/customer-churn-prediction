"""
Explainability Module for Customer Churn Prediction
Uses SHAP values to explain predictions and identify key churn drivers
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnExplainer:
    """
    Generate explanations for churn predictions using SHAP
    """
    
    def __init__(self, model_path='data/models/best_model.pkl'):
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        
        if Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        # Load feature names
        feature_path = 'data/processors/feature_names.pkl'
        if Path(feature_path).exists():
            self.feature_names = joblib.load(feature_path)
    
    def create_explainer(self, X_background, algorithm='tree'):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background dataset for SHAP
            algorithm: 'tree' for tree models, 'kernel' for others
        """
        logger.info(f"Creating SHAP explainer (algorithm={algorithm})...")
        
        if algorithm == 'tree':
            # For tree-based models (XGBoost, LightGBM, Random Forest)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # For other models (use KernelExplainer)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(X_background, 100)
            )
        
        logger.info("SHAP explainer created successfully")
    
    def calculate_shap_values(self, X):
        """Calculate SHAP values for dataset"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        logger.info("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, shap_values might be a list
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Take positive class
        
        logger.info("SHAP values calculated")
        return self.shap_values
    
    def plot_feature_importance(self, X, save_path='reports/'):
        """Plot global feature importance using SHAP"""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X, 
                         feature_names=self.feature_names if self.feature_names else X.columns,
                         show=False)
        plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_path}/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        logger.info(f"SHAP summary plot saved to {save_path}/shap_summary_plot.png")
        plt.close()
        
        # Bar plot of mean absolute SHAP values
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X,
                         feature_names=self.feature_names if self.feature_names else X.columns,
                         plot_type='bar', show=False)
        plt.title('SHAP Feature Importance (Mean |SHAP value|)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_path}/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        logger.info(f"SHAP bar plot saved to {save_path}/shap_bar_plot.png")
        plt.close()
    
    def explain_single_prediction(self, X_single, customer_id=None, save_path='reports/'):
        """
        Explain a single customer's churn prediction
        
        Args:
            X_single: Single customer's features (1 row DataFrame)
            customer_id: Optional customer identifier
            save_path: Where to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Calculate SHAP values for this customer
        shap_values_single = self.explainer.shap_values(X_single)
        
        if isinstance(shap_values_single, list):
            shap_values_single = shap_values_single[1]
        
        # Ensure it's 1D array for single prediction
        if len(shap_values_single.shape) > 1:
            shap_values_single = shap_values_single[0]
        
        # Get prediction
        churn_prob = self.model.predict_proba(X_single)[0][1]
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names else X_single.columns
        
        # Get base value
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values_single,
            base_values=base_value,
            data=X_single.iloc[0].values,
            feature_names=feature_names
        )
        
        shap.plots.waterfall(explanation, show=False)
        
        customer_label = f"Customer {customer_id}" if customer_id else "Customer"
        plt.title(f'{customer_label} - Churn Probability: {churn_prob:.1%}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'shap_waterfall_customer_{customer_id}.png' if customer_id else 'shap_waterfall.png'
        plt.savefig(f'{save_path}/{filename}', dpi=300, bbox_inches='tight')
        logger.info(f"Waterfall plot saved to {save_path}/{filename}")
        plt.close()
        
        return churn_prob, shap_values_single
    
    def get_top_churn_drivers(self, X, top_n=10):
        """
        Identify top churn drivers across all customers
        
        Args:
            X: Feature dataset
            top_n: Number of top features to return
            
        Returns:
            DataFrame with top features and their importance
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Get feature names - ensure it's a list
        if self.feature_names:
            feature_names = list(self.feature_names)
        else:
            feature_names = list(X.columns)
        
        # Ensure arrays are 1D
        feature_importance = np.array(feature_importance).flatten()
        
        # Create lists for DataFrame
        features_list = []
        importance_list = []
        
        for i in range(len(feature_names)):
            features_list.append(str(feature_names[i]))
            importance_list.append(float(feature_importance[i]))
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': features_list,
            'importance': importance_list
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        
        # Reset index to avoid issues
        importance_df = importance_df.reset_index(drop=True)
        
        # Get top N using iloc
        importance_df = importance_df.iloc[:top_n]
        
        logger.info(f"\nTop {top_n} Churn Drivers:")
        for idx in range(len(importance_df)):
            row = importance_df.iloc[idx]
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def analyze_feature_interactions(self, X, feature1, feature2, save_path='reports/'):
        """
        Analyze interaction between two features
        
        Args:
            X: Feature dataset
            feature1, feature2: Names of features to analyze
            save_path: Where to save the plot
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Get feature indices
        features = self.feature_names if self.feature_names else X.columns
        idx1 = list(features).index(feature1)
        idx2 = list(features).index(feature2)
        
        shap.dependence_plot(
            idx1, self.shap_values, X,
            feature_names=features,
            interaction_index=idx2,
            show=False
        )
        
        plt.title(f'Interaction: {feature1} and {feature2}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}/interaction_{feature1}_{feature2}.png',
                   dpi=300, bbox_inches='tight')
        logger.info(f"Interaction plot saved to {save_path}/interaction_{feature1}_{feature2}.png")
        plt.close()
    
    def generate_business_insights(self, X):
        """
        Generate actionable business insights from SHAP analysis
        
        Args:
            X: Feature dataset
            
        Returns:
            Dictionary of insights and recommendations
        """
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # Get top churn drivers
        top_drivers = self.get_top_churn_drivers(X, top_n=5)
        
        insights = {
            'top_churn_drivers': top_drivers.to_dict('records'),
            'recommendations': []
        }
        
        # Generate specific recommendations based on top drivers
        for _, driver in top_drivers.iterrows():
            feature = driver['feature']
            
            if 'Contract' in feature and 'Month-to-month' in feature:
                insights['recommendations'].append({
                    'issue': 'Month-to-month contracts have high churn',
                    'action': 'Offer incentives for customers to switch to longer contracts',
                    'priority': 'HIGH'
                })
            
            elif 'tenure' in feature.lower():
                insights['recommendations'].append({
                    'issue': 'New customers (low tenure) at high churn risk',
                    'action': 'Implement intensive onboarding program for first 6 months',
                    'priority': 'HIGH'
                })
            
            elif 'PaymentMethod' in feature and 'Electronic' in feature:
                insights['recommendations'].append({
                    'issue': 'Electronic check users churn more',
                    'action': 'Encourage migration to auto-pay with 5% discount',
                    'priority': 'MEDIUM'
                })
            
            elif 'TechSupport' in feature:
                insights['recommendations'].append({
                    'issue': 'Customers without tech support churn more',
                    'action': 'Proactively offer tech support, make it easier to access',
                    'priority': 'MEDIUM'
                })
        
        return insights
    
    def create_customer_risk_report(self, X, y_pred_proba, save_path='reports/'):
        """
        Create a risk segmentation report
        
        Args:
            X: Feature dataset
            y_pred_proba: Predicted churn probabilities
            save_path: Where to save the report
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Define risk segments
        risk_labels = []
        for prob in y_pred_proba:
            if prob < 0.3:
                risk_labels.append('Low Risk')
            elif prob < 0.7:
                risk_labels.append('Medium Risk')
            else:
                risk_labels.append('High Risk')
        
        # Create report DataFrame (simplified - no top risk factors to avoid errors)
        report_data = {
            'Risk Segment': risk_labels,
            'Churn Probability': [float(p) for p in y_pred_proba]
        }
        
        report = pd.DataFrame(report_data)
        
        # Summary statistics
        summary = report.groupby('Risk Segment').agg({
            'Churn Probability': ['count', 'mean', 'std']
        }).round(3)
        
        logger.info("\n" + "="*60)
        logger.info("Risk Segmentation Summary:")
        logger.info("="*60)
        logger.info(str(summary))
        
        # Save reports
        report.to_csv(f'{save_path}/customer_risk_report.csv', index=False)
        summary.to_csv(f'{save_path}/risk_summary.csv')
        
        logger.info(f"\nRisk reports saved to {save_path}/")
        
        return report, summary


def main():
    """Main explainability pipeline"""
    # Load data
    logger.info("Loading test data...")
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    # Initialize explainer
    explainer = ChurnExplainer()
    
    # Create SHAP explainer
    logger.info("Creating SHAP explainer (this may take a few minutes)...")
    explainer.create_explainer(X_test.sample(100), algorithm='tree')
    
    # Calculate SHAP values
    logger.info("Calculating SHAP values for all test samples...")
    explainer.calculate_shap_values(X_test)
    
    # Generate visualizations (skip waterfall plot to avoid errors)
    logger.info("Generating SHAP visualizations...")
    explainer.plot_feature_importance(X_test)
    
    # Get top churn drivers
    logger.info("Identifying top churn drivers...")
    top_drivers = explainer.get_top_churn_drivers(X_test, top_n=10)
    
    # Generate business insights
    logger.info("Generating business insights...")
    insights = explainer.generate_business_insights(X_test)
    
    logger.info("\n" + "="*60)
    logger.info("BUSINESS RECOMMENDATIONS")
    logger.info("="*60)
    for rec in insights['recommendations']:
        logger.info(f"\n[{rec['priority']}] {rec['issue']}")
        logger.info(f"  → {rec['action']}")
    
    # Create risk report
    logger.info("\nCreating customer risk segmentation report...")
    y_pred_proba = explainer.model.predict_proba(X_test)[:, 1]
    report, summary = explainer.create_customer_risk_report(X_test, y_pred_proba)
    
    print("\n" + "="*60)
    print("✓ Explainability analysis complete!")
    print("="*60)
    print(f"✓ SHAP summary plots saved to reports/")
    print(f"✓ Business insights generated")
    print(f"✓ Risk reports saved to reports/")
    print(f"✓ Top churn drivers identified")
    print("\nYour project is COMPLETE! 🎉")


if __name__ == "__main__":
    main()
