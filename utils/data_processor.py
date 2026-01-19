import pandas as pd
from datetime import datetime, timedelta
import random

def generate_sample_data(crop_name, db, days=730):
    """
    Generate sample historical price data for testing
    Only use this if you don't have real data
    """
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(days, 0, -1)]
    
    # Base price for different crops
    base_prices = {
        'Rice': 2000,
        'Wheat': 1800,
        'Tomato': 1500,
        'Cotton': 5000,
        'Sugarcane': 3000,
        'Potato': 1200
    }
    
    base_price = base_prices.get(crop_name, 2000)
    
    # Generate prices with trend and seasonality
    prices = []
    for i, date in enumerate(dates):
        # Add trend
        trend = (i / days) * 200
        # Add seasonality (yearly cycle)
        seasonality = 300 * np.sin(2 * np.pi * i / 365)
        # Add random noise
        noise = random.uniform(-100, 100)
        
        price = base_price + trend + seasonality + noise
        prices.append({
            'crop': crop_name,
            'date': date,
            'price': max(price, base_price * 0.5),  # Ensure positive prices
            'location': 'Varanasi',
            'unit': 'quintal',
            'coordinates': {
                'type': 'Point',
                'coordinates': [82.9739, 25.3176]  # Varanasi coordinates
            }
        })
    
    # Insert into database
    try:
        db.marketprices.insert_many(prices)
        print(f"Inserted {len(prices)} sample price records for {crop_name}")
        return True
    except Exception as e:
        print(f"Error inserting sample data: {e}")
        return False

if __name__ == '__main__':
    from pymongo import MongoClient
    from dotenv import load_dotenv
    import os
    import numpy as np
    
    load_dotenv()
    
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    client = MongoClient(MONGO_URI)
    db = client['farm_db']
    
    # Generate sample data for all crops
    crops = ['Rice', 'Wheat', 'Tomato', 'Cotton', 'Sugarcane', 'Potato']
    
    for crop in crops:
        generate_sample_data(crop, db, days=730)