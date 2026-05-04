"""
FastAPI Application for Customer Churn Prediction
Production-ready API with authentication and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="AI-powered churn prediction with explainable insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Simple API key authentication (in production, use proper auth)
API_KEY = "your-secret-api-key-here"


# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    """Customer features for prediction"""
    tenure: int = Field(..., description="Months with company", ge=0, le=100)
    MonthlyCharges: float = Field(..., description="Monthly charges", ge=0, le=200)
    TotalCharges: float = Field(..., description="Total charges", ge=0)
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method type")
    
    class Config:
        schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00,
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    customer_id: Optional[str] = None
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: str = Field(..., description="Yes or No")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    confidence: float = Field(..., description="Model confidence")
    top_risk_factors: List[Dict[str, float]] = Field(..., description="Top 3 features driving churn risk")
    recommendation: str = Field(..., description="Action recommendation")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    average_churn_probability: float


# Global variables for model and processors
model = None
scaler = None
label_encoders = None
feature_names = None
processor = None


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials


def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    global model, scaler, label_encoders, feature_names, processor
    
    try:
        # Load model
        model_path = 'data/models/best_model.pkl'
        if Path(model_path).exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model not found at {model_path}")
        
        # Load processors
        scaler = joblib.load('data/processors/scaler.pkl')
        label_encoders = joblib.load('data/processors/label_encoders.pkl')
        feature_names = joblib.load('data/processors/feature_names.pkl')
        
        # Import processor
        from data_processing import ChurnDataProcessor
        processor = ChurnDataProcessor()
        processor.scaler = scaler
        processor.label_encoders = label_encoders
        processor.feature_names = feature_names
        
        logger.info("All artifacts loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")
        raise


def preprocess_customer_data(customer: CustomerFeatures) -> pd.DataFrame:
    """Preprocess customer data for prediction"""
    # Convert to DataFrame
    data = pd.DataFrame([customer.dict()])
    
    # Feature engineering
    data = processor.engineer_features(data)
    
    # Encoding
    data_encoded = processor.encode_features(data, fit=False)
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in data_encoded.columns:
            data_encoded[feature] = 0
    
    # Select and order features
    data_encoded = data_encoded[feature_names]
    
    # Scale
    data_scaled = processor.scale_features(data_encoded, fit=False)
    
    return data_scaled


def get_top_risk_factors(customer_data: pd.DataFrame, n_top=3) -> List[Dict[str, float]]:
    """Get top risk factors for a customer"""
    # Get feature importance from model (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get indices of top N features
        top_indices = np.argsort(importances)[-n_top:][::-1]
        
        risk_factors = []
        for idx in top_indices:
            risk_factors.append({
                'feature': feature_names[idx],
                'importance': float(importances[idx]),
                'customer_value': float(customer_data.iloc[0, idx])
            })
        
        return risk_factors
    else:
        return []


def get_recommendation(churn_prob: float, risk_factors: List) -> str:
    """Generate action recommendation based on prediction"""
    if churn_prob < 0.3:
        return "Customer at low risk. Continue current engagement strategy."
    
    elif churn_prob < 0.7:
        recommendations = [
            "Schedule a check-in call to address any concerns.",
            "Offer a loyalty reward or service upgrade.",
            "Provide personalized recommendations based on usage."
        ]
        
        # Customize based on risk factors
        if risk_factors:
            top_factor = risk_factors[0]['feature']
            if 'Contract' in top_factor:
                return "URGENT: Offer contract upgrade incentive (10-15% discount for longer term)."
            elif 'tenure' in top_factor.lower():
                return "URGENT: Initiate retention campaign - customer is in high-risk early period."
            elif 'TechSupport' in top_factor:
                return "URGENT: Proactively offer technical support consultation."
        
        return recommendations[0]
    
    else:
        return "HIGH PRIORITY: Immediate intervention required. Assign to retention specialist within 24 hours."


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting API...")
    load_model_artifacts()
    logger.info("API ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": label_encoders is not None,
        "feature_count": len(feature_names) if feature_names else 0,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(
    customer: CustomerFeatures,
    customer_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict churn for a single customer
    
    Args:
        customer: Customer features
        customer_id: Optional customer identifier
        
    Returns:
        Prediction with probability, risk level, and recommendations
    """
    try:
        # Preprocess data
        customer_data = preprocess_customer_data(customer)
        
        # Make prediction
        churn_prob = model.predict_proba(customer_data)[0][1]
        churn_prediction = "Yes" if churn_prob > 0.5 else "No"
        
        # Determine risk level
        if churn_prob < 0.3:
            risk_level = "LOW"
        elif churn_prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Get confidence (distance from decision boundary)
        confidence = abs(churn_prob - 0.5) * 2
        
        # Get top risk factors
        risk_factors = get_top_risk_factors(customer_data)
        
        # Get recommendation
        recommendation = get_recommendation(churn_prob, risk_factors)
        
        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=float(churn_prob),
            churn_prediction=churn_prediction,
            risk_level=risk_level,
            confidence=float(confidence),
            top_risk_factors=risk_factors,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict churn for multiple customers
    
    Args:
        request: Batch prediction request with list of customers
        
    Returns:
        Batch prediction response with all predictions
    """
    try:
        predictions = []
        high_risk_count = 0
        total_prob = 0
        
        for idx, customer in enumerate(request.customers):
            # Make individual prediction
            pred = await predict_churn(customer, customer_id=f"BATCH_{idx}")
            predictions.append(pred)
            
            if pred.risk_level == "HIGH":
                high_risk_count += 1
            
            total_prob += pred.churn_probability
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count,
            average_churn_probability=total_prob / len(predictions) if predictions else 0
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    """Get model information"""
    try:
        info_path = 'data/models/model_info.pkl'
        if Path(info_path).exists():
            info = joblib.load(info_path)
            return info
        else:
            return {"message": "Model info not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
async def get_features():
    """Get list of features used by the model"""
    if feature_names:
        return {
            "features": feature_names,
            "count": len(feature_names)
        }
    else:
        raise HTTPException(status_code=404, detail="Feature names not loaded")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
