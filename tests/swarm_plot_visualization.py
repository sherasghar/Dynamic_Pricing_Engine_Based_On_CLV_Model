import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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

def generate_segmented_data(num_samples=150):
    """Generate test data with natural segments and unique indices"""
    np.random.seed(42)
    
    # Create three customer segments with unique IDs
    segments = {
        'Low-Value': {'multiplier': 0.5, 'count': int(num_samples*0.4)},
        'Mid-Value': {'multiplier': 1.0, 'count': int(num_samples*0.35)},
        'High-Value': {'multiplier': 2.0, 'count': int(num_samples*0.25)}
    }
    
    data = []
    id_counter = 0
    for segment, params in segments.items():
        segment_data = {
            'CustomerID': range(id_counter, id_counter + params['count']),
            'Segment': segment,
            'Recency': np.random.randint(1, 365, params['count']),
            'Frequency': np.maximum(1, np.random.poisson(5 * params['multiplier'], params['count'])),
            'MonetaryValue': np.random.uniform(50, 5000, params['count']) * params['multiplier'],
            'Tenure': np.random.randint(30, 365*5, params['count']),
            'AvgDaysBetweenPurchases': np.random.randint(7, 90, params['count']),
            'Age': np.random.randint(18, 80, params['count']),
            'UniqueProductsCount': np.random.randint(1, 10, params['count'])
        }
        id_counter += params['count']
        data.append(pd.DataFrame(segment_data))
    
    df = pd.concat(data)
    df.set_index('CustomerID', inplace=True)
    return df

def create_swarm_plots():
    """Create and save optimized swarm plot visualization"""
    # Generate segmented test data with unique indices
    df = generate_segmented_data(120)  # Reduced sample size for better swarm display
    
    # Get model predictions
    clv_values = []
    price_values = []
    
    for _, row in df.iterrows():
        customer_data = row.to_dict()
        customer_data['product_cost'] = 50.0
        
        try:
            response = client.post("/api/calculate_price/", json=customer_data)
            if response.status_code == 200:
                result = response.json()['data']
                clv_values.append(result['clv'])
                price_values.append(result['dynamic_price'])
            else:
                clv_values.append(np.nan)
                price_values.append(np.nan)
        except:
            clv_values.append(np.nan)
            price_values.append(np.nan)
    
    df['Predicted_CLV'] = clv_values
    df['Dynamic_Price'] = price_values
    df['Price_CLV_Ratio'] = df['Dynamic_Price'] / df['Predicted_CLV']
    
    # Clean data and ensure unique indices
    df = df.dropna(subset=['Predicted_CLV', 'Dynamic_Price'])
    df = df.reset_index().drop_duplicates(subset='CustomerID').set_index('CustomerID')
    
    # Set up figure with adjusted layout
    plt.figure(figsize=(16, 12))
    sns.set(style="whitegrid", palette="pastel", font_scale=1.1)
    
    # 1. CLV Distribution by Segment (using stripplot with jitter)
    plt.subplot(2, 2, 1)
    sns.stripplot(x='Segment', y='Predicted_CLV', data=df, 
                 order=['Low-Value', 'Mid-Value', 'High-Value'],
                 size=6, hue='Frequency', palette='coolwarm', 
                 jitter=0.25, alpha=0.7, dodge=True)
    plt.title('CLV Distribution by Customer Segment', pad=20)
    plt.xlabel('')
    plt.ylabel('Predicted CLV ($)')
    plt.legend(title='Purchase Frequency', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # 2. Price Distribution by Segment (swarmplot with reduced size)
    plt.subplot(2, 2, 2)
    sns.swarmplot(x='Segment', y='Dynamic_Price', data=df,
                 order=['Low-Value', 'Mid-Value', 'High-Value'],
                 size=3, hue='Recency', palette='viridis', warn_thresh=0.1)
    plt.title('Dynamic Price Distribution by Segment', pad=20)
    plt.xlabel('')
    plt.ylabel('Dynamic Price ($)')
    plt.legend(title='Recency (days)', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # 3. Price/CLV Ratio by Segment (boxplot with stripplot overlay)
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Segment', y='Price_CLV_Ratio', data=df,
               order=['Low-Value', 'Mid-Value', 'High-Value'],
               showfliers=False, width=0.4)
    sns.stripplot(x='Segment', y='Price_CLV_Ratio', data=df,
                 order=['Low-Value', 'Mid-Value', 'High-Value'],
                 size=4, color='black', alpha=0.5, jitter=0.2)
    plt.title('Price-to-CLV Ratio by Segment', pad=20)
    plt.xlabel('Customer Segment')
    plt.ylabel('Price/CLV Ratio')
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Monetary Value vs CLV with segments
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='MonetaryValue', y='Predicted_CLV', data=df,
                   hue='Segment', style='Segment',
                   palette='Set2', s=100, alpha=0.7)
    plt.title('Monetary Value vs CLV', pad=20)
    plt.xlabel('Monetary Value ($)')
    plt.ylabel('Predicted CLV ($)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Final formatting
    plt.tight_layout(pad=3.0)
    
    # Save output
    os.makedirs("test_results", exist_ok=True)
    output_path = "test_results/clv_swarm_analysis.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Optimized swarm plot analysis saved to {output_path}")
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
    output_file = create_swarm_plots()
    if not os.path.exists(output_file):
        print("Error: Visualization failed to generate")
        sys.exit(1)
    
    print("CLV swarm plot analysis completed successfully")