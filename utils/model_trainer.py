import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = 'models'

def ensure_model_dir():
    """Ensure models directory exists"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

def train_crop_model(crop_name, db, days=1200):
    """
    Train a Prophet model for a specific crop
    
    Args:
        crop_name: Name of the crop
        db: MongoDB database instance
        days: Number of days of historical data to use (default 2 years)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        ensure_model_dir()
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        prices = list(db.historicalprices.find({
            'crop': crop_name,
            'date': {'$gte': start_date, '$lte': end_date}
        }).sort('date', 1))
        print(prices)
        print(crop_name)
        
        if len(prices) < 100:  # Minimum data requirement
            return False, f"Insufficient data for {crop_name}. Need at least 100 records, found {len(prices)}"
        
        # Prepare data for Prophet
        df = pd.DataFrame(prices)
        df = df[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
        
        # Initialize and train Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        model.fit(df)
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, f'{crop_name}_model.pkl')
        joblib.dump(model, model_path)
        
        logger.info(f"Successfully trained and saved model for {crop_name}")
        return True, f"Model trained successfully with {len(prices)} data points"
        
    except Exception as e:
        logger.error(f"Error training model for {crop_name}: {e}")
        return False, str(e)

def train_all_crops(db):
    """Train models for all crops that have sufficient data"""
    crops = db.historicalprices.distinct('crop')
    results = {}
    
    for crop in crops:
        success, message = train_crop_model(crop, db)
        results[crop] = {'success': success, 'message': message}
        print(f"{crop}: {message}")
    
    return results

if __name__ == '__main__':
    # For standalone training
    from pymongo import MongoClient
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    client = MongoClient(MONGO_URI)
    db = client['test']
    
    print("Starting model training...")
    results = train_all_crops(db)
    
    print("\n=== Training Summary ===")
    for crop, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {crop}: {result['message']}")