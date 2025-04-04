import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from fastapi.testclient import TestClient

# Set up paths and imports
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

try:
    from api.app import app
    from api.pricing_engine import PricingEngine
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure:")
    print("1. You're running from the project root directory")
    print("2. Your API module is properly structured")
    sys.exit(1)

client = TestClient(app)

def generate_test_data(num_samples=100):
    """Generate synthetic test data with realistic ranges"""
    data = {
        'Recency': np.random.randint(1, 365, num_samples),
        'Frequency': np.random.randint(1, 50, num_samples),
        'MonetaryValue': np.random.uniform(50, 5000, num_samples),
        'Tenure': np.random.randint(30, 365*5, num_samples),
        'AvgDaysBetweenPurchases': np.random.randint(1, 90, num_samples),
        'Age': np.random.randint(18, 80, num_samples),
        'UniqueProductsCount': np.random.randint(1, 10, num_samples)
    }
    return pd.DataFrame(data)

def create_clv_scatter_plot():
    """Create and save scatter plot of CLV relationships"""
    # Generate test data
    df = generate_test_data(50)
    
    # Get predictions
    clv_predictions = []
    price_predictions = []
    
    for _, row in df.iterrows():
        customer_data = row.to_dict()
        customer_data['product_cost'] = 50.0  # Add required field
        
        try:
            response = client.post("/api/calculate_price/", json=customer_data)
            if response.status_code == 200:
                result = response.json()['data']
                clv_predictions.append(result['clv'])
                price_predictions.append(result['dynamic_price'])
            else:
                print(f"Error with customer {_}: {response.text}")
                clv_predictions.append(np.nan)
                price_predictions.append(np.nan)
        except Exception as e:
            print(f"API call failed: {e}")
            clv_predictions.append(np.nan)
            price_predictions.append(np.nan)
    
    df['Predicted_CLV'] = clv_predictions
    df['Dynamic_Price'] = price_predictions
    
    # Create figure with 3 subplots
    plt.figure(figsize=(18, 5))
    
    # 1. CLV vs MonetaryValue
    plt.subplot(1, 3, 1)
    plt.scatter(df['MonetaryValue'], df['Predicted_CLV'], 
               c=df['Frequency'], alpha=0.6, cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.xlabel('Monetary Value ($)')
    plt.ylabel('Predicted CLV ($)')
    plt.title('CLV vs Monetary Value')
    plt.grid(True, alpha=0.3)
    
    # 2. Price vs CLV
    plt.subplot(1, 3, 2)
    plt.scatter(df['Predicted_CLV'], df['Dynamic_Price'], 
               c=df['Recency'], alpha=0.6, cmap='plasma')
    plt.colorbar(label='Recency (days)')
    plt.xlabel('Predicted CLV ($)')
    plt.ylabel('Dynamic Price ($)')
    plt.title('Price vs CLV')
    plt.grid(True, alpha=0.3)
    
    # 3. Price Ratio vs Tenure
    price_ratio = df['Dynamic_Price'] / df['Predicted_CLV']
    plt.subplot(1, 3, 3)
    plt.scatter(df['Tenure'], price_ratio, 
               c=df['Age'], alpha=0.6, cmap='cool')
    plt.colorbar(label='Customer Age')
    plt.xlabel('Tenure (days)')
    plt.ylabel('Price/CLV Ratio')
    plt.title('Pricing Ratio vs Customer Tenure')
    plt.grid(True, alpha=0.3)
    
    # Final formatting
    plt.tight_layout()
    
    # Save output
    os.makedirs("test_results", exist_ok=True)
    output_path = "test_results/clv_scatter_analysis.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Scatter plot analysis saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Verify API connection
    try:
        response = client.get("/health")
        if response.status_code != 200:
            print("API health check failed")
            sys.exit(1)
    except Exception as e:
        print(f"API connection failed: {e}")
        sys.exit(1)
    
    # Generate and verify visualization
    output_file = create_clv_scatter_plot()
    if not os.path.exists(output_file):
        print("Error: Visualization failed to generate")
        sys.exit(1)
    
    print("CLV scatter plot analysis completed successfully")