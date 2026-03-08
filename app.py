from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client[os.getenv('DB_NAME', 'farm_db')]

# Load trained models on startup
MODELS = {}
MODEL_DIR = 'models'

def load_models():
    """Load all trained models into memory"""
    global MODELS
    if not os.path.exists(MODEL_DIR):
        logger.warning(f"Model directory {MODEL_DIR} not found. Train models first.")
        return
    
    for filename in os.listdir(MODEL_DIR):
        if filename.endswith('_model.pkl'):
            crop_district = filename.replace('_model.pkl', '')
            model_path = os.path.join(MODEL_DIR, filename)
            try:
                MODELS[crop_district] = joblib.load(model_path)
                logger.info(f"Loaded model for {crop_district}")
            except Exception as e:
                logger.error(f"Failed to load model for {crop_district}: {e}")

# Load models on startup
load_models()

def get_historical_data(crop_name, district='Varanasi', days=1300):
    """Fetch historical price data from MongoDB HistoricalPrice collection"""
    try:
        # Fetch from historicalprices collection
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        prices = list(db.historicalprices.find({
            'crop': crop_name,
            'district': district,
            'date': {'$gte': start_date, '$lte': end_date}
        }).sort('date', 1))
        
        if not prices:
            logger.warning(f"No historical data found for {crop_name} in {district}")
            return None
        
        df = pd.DataFrame(prices)
        df = df[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
        
        # Ensure datetime format
        df['ds'] = pd.to_datetime(df['ds'])
        
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

def calculate_market_factors(crop_name, historical_data):
    """Calculate market factors based on historical data"""
    try:
        recent_prices = historical_data['y'].tail(30).values
        older_prices = historical_data['y'].tail(60).head(30).values
        
        recent_avg = np.mean(recent_prices)
        older_avg = np.mean(older_prices)
        volatility = np.std(recent_prices) / recent_avg * 100
        
        # Determine trend
        if recent_avg > older_avg * 1.05:
            seasonal_trend = 'increasing'
        elif recent_avg < older_avg * 0.95:
            seasonal_trend = 'decreasing'
        else:
            seasonal_trend = 'stable'
        
        # Define market factors
        factors = [
            {
                'name': 'Seasonal Demand',
                'impact': 'high' if volatility > 15 else 'medium',
                'trend': 'up' if seasonal_trend == 'increasing' else 'down' if seasonal_trend == 'decreasing' else 'stable'
            },
            {
                'name': 'Supply Chain',
                'impact': 'medium',
                'trend': 'stable'
            },
            {
                'name': 'Weather Conditions',
                'impact': 'high',
                'trend': 'up'
            },
            {
                'name': 'Government Policies',
                'impact': 'low',
                'trend': 'stable'
            }
        ]
        
        return factors, seasonal_trend
    except Exception as e:
        logger.error(f"Error calculating market factors: {e}")
        return [], 'stable'

def make_prediction(crop_name, district, timeframe):
    """Make price prediction using trained model"""
    try:
        # Create model key
        model_key = f"{crop_name}_{district}"
        
        # Check if model exists
        if model_key not in MODELS:
            logger.warning(f"No trained model found for {model_key}")
            return None, f"No trained model found for {crop_name} in {district}"
        
        model = MODELS[model_key]
        
        # Get historical data
        historical_data = get_historical_data(crop_name, district, days=730)
        if historical_data is None or len(historical_data) < 30:
            return None, "Insufficient historical data"
        
        # Determine prediction days
        days_map = {
            '1month': 30,
            '3months': 90,
            '6months': 180
        }
        days_ahead = days_map.get(timeframe, 90)
        print(days_ahead)
        current_ahead = 0
        # Make prediction using Prophet
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        current = model.make_future_dataframe(periods=current_ahead)
        forecast_current = model.predict(current)
        
        # Get current and predicted prices
        current_price =forecast_current['yhat'].iloc[-1]
        print(current_price)
        predicted_price = forecast['yhat'].iloc[-1]
        
        # Calculate change
        change = predicted_price - current_price
        change_percent = (change / current_price) * 100
        
        # Calculate confidence based on prediction interval width
        prediction_interval = forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]
        confidence = max(40, min(95, 100 - (prediction_interval / predicted_price * 50)))
        
        # Calculate historical average
        historical_avg = historical_data['y'].mean()
        
        # Get market factors
        factors, seasonal_trend = calculate_market_factors(crop_name, historical_data)
        
        # Prepare chart data (last 60 days + prediction)
        chart_data = []
        historical_chart = historical_data.tail(60)
        
        for idx, row in historical_chart.iterrows():
            chart_data.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'price': float(row['y']),
                'type': 'historical'
            })
        
        # Add predicted point
        future_date = historical_data['ds'].iloc[-1] + timedelta(days=days_ahead)
        chart_data.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'price': float(predicted_price),
            'type': 'predicted'
        })
        
        result = {
            'currentPrice': float(current_price),
            'predictedPrice': float(predicted_price),
            'change': float(change),
            'changePercent': round(change_percent, 2),
            'confidence': round(confidence, 1),
            'historicalAvg': float(historical_avg),
            'seasonalTrend': seasonal_trend,
            'chartData': chart_data,
            'factors': factors
        }
        
        return result, None
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS),
        'available_models': list(MODELS.keys())
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.json
        crop = data.get('crop')
        district = data.get('district', 'Varanasi')
        timeframe = data.get('timeframe', '3months')
        
        if not crop:
            return jsonify({'error': 'Crop name is required'}), 400
        
        if timeframe not in ['1month', '3months', '6months']:
            return jsonify({'error': 'Invalid timeframe. Use 1month, 3months, or 6months'}), 400
        
        result, error = make_prediction(crop, district, timeframe)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'crop': crop,
            'district': district,
            'timeframe': timeframe,
            'forecast': result
        }), 200
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/available-crops', methods=['GET'])
def available_crops():
    """Get list of crops with trained models"""
    crops_districts = {}
    for key in MODELS.keys():
        crop, district = key.rsplit('_', 1)
        if crop not in crops_districts:
            crops_districts[crop] = []
        crops_districts[crop].append(district)
    
    return jsonify({
        'crops': crops_districts
    }), 200

@app.route('/retrain/<crop>/<district>', methods=['POST'])
def retrain_model(crop, district):
    """Retrain model for a specific crop and district (admin only)"""
    try:
        from utils.model_trainer import train_crop_model
        
        success, message = train_crop_model(crop, district, db)
        
        if success:
            # Reload the model
            model_key = f"{crop}_{district}"
            model_path = os.path.join(MODEL_DIR, f'{model_key}_model.pkl')
            MODELS[model_key] = joblib.load(model_path)
            return jsonify({'success': True, 'message': message}), 200
        else:
            return jsonify({'success': False, 'error': message}), 500
            
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7000))
    app.run(host='0.0.0.0', port=port, debug=False)
