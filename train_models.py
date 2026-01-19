#!/usr/bin/env python3
"""
One-time script to train all crop price prediction models
Run this before starting the Flask service
"""

from pymongo import MongoClient
from dotenv import load_dotenv
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_trainer import train_all_crops

def main():
    load_dotenv()
    
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    DB_NAME = os.getenv('DB_NAME', 'farm_db')
    print(f"Connecting to MongoDB at: {MONGO_URI}")
    print(f"Database: {DB_NAME}")
    
    try:
        client = MongoClient(MONGO_URI)
        db = client['test']
        
        # Test connection
        client.server_info()
        print("✓ Connected to MongoDB successfully")
        doc = db.historicalprices.find_one()
        print(doc)

        print(client.list_database_names())

        
        # Check available data
        pipeline = [
            {
                '$group': {
                    '_id': {
                        'crop': '$crop',
                        'district': '$district'
                    },
                    'count': {'$sum': 1}
                }
            }
        ]
        combinations = list(db.historicalprices.aggregate(pipeline))
        
        
        print(f"\nFound {len(combinations)} crop-district combinations:")
        for combo in combinations:
            crop = combo['_id']['crop']
            district = combo['_id']['district']
            count = combo['count']
            print(f"  - {crop} ({district}): {count} records")
        
        # Train models
        print("\n" + "="*50)
        print("Starting model training...")
        print("="*50 + "\n")
        
        results = train_all_crops(db)
        
        # Print summary
        print("\n" + "="*50)
        print("Training Complete - Summary")
        print("="*50)
        
        successful = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        print(f"\nSuccessfully trained: {successful}/{total} models\n")
        
        for key, result in results.items():
            status = "✓" if result['success'] else "✗"
            print(f"{status} {key}: {result['message']}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()