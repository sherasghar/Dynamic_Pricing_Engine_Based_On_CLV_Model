import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from fastapi.testclient import TestClient

# Set up paths and imports
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

try:
    from api.app import app
    from api.pricing_engine import PricingEngine
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

client = TestClient(app)

def load_test_data():
    """Load or generate test data with actual and predicted values"""
    # Replace this with your actual test dataset loading logic
    # This example generates synthetic data for demonstration
    np.random.seed(42)
    n_samples = 200
    
    # Simulated features (adjust ranges to match your actual data)
    X = {
        'Recency': np.random.randint(1, 365, n_samples),
        'Frequency': np.random.randint(1, 50, n_samples),
        'MonetaryValue': np.random.uniform(50, 5000, n_samples),
        'Tenure': np.random.randint(30, 365*5, n_samples),
        'AvgDaysBetweenPurchases': np.random.randint(7, 90, n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'UniqueProductsCount': np.random.randint(1, 10, n_samples)
    }
    df = pd.DataFrame(X)
    
    # Simulate actual CLV values (replace with your actual targets)
    df['Actual_CLV'] = (
        0.3 * df['MonetaryValue'] +
        0.2 * df['Frequency'] * 100 +
        0.1 * (365 - df['Recency']) +
        np.random.normal(0, 100, n_samples)
    )
    
    return df

def evaluate_model_r2():
    """Calculate and visualize R² score for the CLV model"""
    df = load_test_data()
    
    # Get model predictions
    actual_values = []
    predicted_values = []
    
    for _, row in df.iterrows():
        customer_data = row.to_dict()
        customer_data['product_cost'] = 50.0  # Add required field
        
        try:
            response = client.post("/api/calculate_price/", json=customer_data)
            if response.status_code == 200:
                result = response.json()['data']
                actual_values.append(row['Actual_CLV'])
                predicted_values.append(result['clv'])
        except Exception as e:
            print(f"Skipping customer {_}: {str(e)}")
            continue
    
    # Calculate R² score
    r2 = r2_score(actual_values, predicted_values)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Actual vs Predicted scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(actual_values, predicted_values, alpha=0.6)
    plt.plot([min(actual_values), max(actual_values)], 
             [min(actual_values), max(actual_values)], 
             'r--', linewidth=2)
    plt.xlabel('Actual CLV ($)')
    plt.ylabel('Predicted CLV ($)')
    plt.title(f'Actual vs Predicted CLV\nR² = {r2:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    residuals = np.array(actual_values) - np.array(predicted_values)
    plt.subplot(1, 2, 2)
    plt.scatter(predicted_values, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted CLV ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    os.makedirs("model_evaluation", exist_ok=True)
    plot_path = "model_evaluation/clv_r2_evaluation.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save numerical results
    results = {
        "r2_score": r2,
        "n_samples": len(actual_values),
        "mean_absolute_error": np.mean(np.abs(residuals)),
        "root_mean_squared_error": np.sqrt(np.mean(residuals**2))
    }
    
    return results, plot_path

if __name__ == "__main__":
    # Verify API connection
    try:
        health_check = client.get("/health")
        if health_check.status_code != 200:
            print("API health check failed")
            sys.exit(1)
    except Exception as e:
        print(f"API connection failed: {e}")
        sys.exit(1)
    
    # Run evaluation
    metrics, plot_file = evaluate_model_r2()
    
    # Print results
    print("\nCLV Model Evaluation Results:")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Samples Used: {metrics['n_samples']}")
    print(f"Mean Absolute Error: ${metrics['mean_absolute_error']:.2f}")
    print(f"RMSE: ${metrics['root_mean_squared_error']:.2f}")
    print(f"\nVisualization saved to: {plot_file}")
    
    # Interpretation guide
    print("\nR² Score Interpretation:")
    print("0.9-1.0: Excellent predictive power")
    print("0.8-0.9: Very good")
    print("0.7-0.8: Good")
    print("0.6-0.7: Moderate")
    print("0.5-0.6: Weak")
    print("<0.5: Poor predictive power")