import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

# Import your app after setting the path
try:
    from api.app import app
except ImportError as e:
    print(f"Error importing API module: {e}")
    print("Please ensure:")
    print("1. You're running from the project root directory")
    print("2. Your API module has an __init__.py file")
    print("3. The directory structure is correct")
    sys.exit(1)

client = TestClient(app)

def generate_test_customer():
    """Generate a test customer with realistic values"""
    return {
        "Recency": np.random.randint(1, 90),
        "Frequency": np.random.randint(1, 20),
        "MonetaryValue": np.random.randint(100, 2000),
        "Tenure": np.random.randint(30, 365*3),
        "AvgDaysBetweenPurchases": np.random.randint(7, 60),
        "Age": np.random.randint(18, 70),
        "UniqueProductsCount": np.random.randint(1, 8),
        "product_cost": 50.0
    }

def test_and_visualize_price_comparison():
    """Test and visualize CLV vs Final Price comparison"""
    # Generate test data
    test_customers = [generate_test_customer() for _ in range(5)]
    results = []
    
    for customer in test_customers:
        response = client.post("/api/calculate_price/", json=customer)
        if response.status_code != 200:
            print(f"API request failed for customer: {response.text}")
            continue
            
        data = response.json()["data"]
        results.append({
            "customer": customer,
            "clv": data["clv"],
            "final_price": data["dynamic_price"]
        })
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    for i, result in enumerate(results, 1):
        plt.subplot(2, 3, i)
        bars = plt.bar(['Predicted CLV', 'Final Price'], 
                      [result["clv"], result["final_price"]],
                      color=['#1f77b4', '#2ca02c'])
        
        plt.title(f'Customer {i}')
        plt.ylabel('Value ($)')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'${height:.2f}',
                    ha='center', va='bottom')
        
        # Add ratio line
        plt.axhline(y=result["clv"], color='r', linestyle='--', alpha=0.3)
        ratio = result["final_price"] / result["clv"]
        plt.text(0.5, result["clv"]*1.05, f'Ratio: {ratio:.2f}x', 
                ha='center', color='red')

    plt.suptitle('CLV vs Final Price Comparison', y=1.02)
    plt.tight_layout()
    
    # Save results
    os.makedirs("test_results", exist_ok=True)
    output_path = "test_results/clv_price_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Successfully generated chart at {output_path}")
    return output_path

if __name__ == "__main__":
    # Verify we can access the API
    try:
        health_check = client.get("/health")
        if health_check.status_code != 200:
            print("API health check failed")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to connect to API: {e}")
        sys.exit(1)
    
    # Run the test
    output_file = test_and_visualize_price_comparison()
    
    # Verify output was created
    if not os.path.exists(output_file):
        print("Error: Chart was not generated successfully")
        sys.exit(1)
    
    print("Test completed successfully")