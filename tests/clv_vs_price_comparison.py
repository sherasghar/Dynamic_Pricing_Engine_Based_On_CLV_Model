import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import joblib
import os

class CLVVisualizer:
    def __init__(self, model_path, results_path):
        """Initialize with model and results data"""
        # Verify paths exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found at: {results_path}")
            
        try:
            self.model = joblib.load(model_path)
            self.clv_results = pd.read_csv(results_path)
        except Exception as e:
            raise ValueError(f"Error loading files: {str(e)}")
        
        # Verify required columns exist
        required_columns = {'CustomerID', 'Predicted_CLV'}
        if not required_columns.issubset(self.clv_results.columns):
            missing = required_columns - set(self.clv_results.columns)
            raise ValueError(f"CSV missing required columns: {missing}")
        
        # Convert CustomerID to string if not already
        self.clv_results['CustomerID'] = self.clv_results['CustomerID'].astype(str)
        
        print("Data loaded successfully. CLV summary statistics:")
        print(self.clv_results['Predicted_CLV'].describe())
    
    def calculate_dynamic_price(self, clv, base_price=100):
        """Calculate final price based on CLV segment"""
        # Get percentiles
        percentiles = np.percentile(self.clv_results['Predicted_CLV'], [25, 75])
        
        # Segment customers
        if clv > percentiles[1]:  # Above 75th percentile
            segment = 'high'
            adjustment = 0.9  # 10% discount for high-value customers
        elif clv < percentiles[0]:  # Below 25th percentile
            segment = 'low'
            adjustment = 1.1  # 10% premium for low-value customers
        else:
            segment = 'medium'
            adjustment = 1.0  # Standard price
            
        print(f"CLV: ${clv:,.2f} → Segment: {segment} → Price Adjustment: {adjustment}x")
        return base_price * adjustment
    
    def plot_clv_vs_price(self, customer_ids=None, save_path='clv_vs_price_comparison.png'):
        """
        Plot comparison between predicted CLV and final price
        customer_ids: List of specific customers to plot (None for random sample)
        """
        # Select customers to visualize
        if customer_ids:
            # Convert input IDs to strings for comparison
            customer_ids = [str(cid) for cid in customer_ids]
            plot_data = self.clv_results[self.clv_results['CustomerID'].isin(customer_ids)].copy()
            
            if plot_data.empty:
                available_ids = self.clv_results['CustomerID'].head(10).tolist()
                raise ValueError(
                    f"No customers found with IDs: {customer_ids}\n"
                    f"First 10 available IDs: {available_ids}"
                )
        else:
            plot_data = self.clv_results.sample(n=min(5, len(self.clv_results)), random_state=42).copy()
        
        print("\nCustomers being visualized:")
        print(plot_data[['CustomerID', 'Predicted_CLV']].to_string(index=False))
        
        # Calculate final prices
        plot_data['Final_Price'] = plot_data['Predicted_CLV'].apply(
            lambda x: self.calculate_dynamic_price(x))
        
        # Create figure with better proportions
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
        
        # Plot each customer
        bar_width = 0.35
        x_pos = np.arange(len(plot_data))
        
        for idx, (_, row) in enumerate(plot_data.iterrows()):
            ax.bar([idx - bar_width/2, idx + bar_width/2],
                   [row['Predicted_CLV'], row['Final_Price']],
                   width=bar_width,
                   color=colors[idx],
                   alpha=0.7,
                   edgecolor='black',
                   label=f"Customer {row['CustomerID']}")
        
        # Formatting
        max_value = max(plot_data['Predicted_CLV'].max(), plot_data['Final_Price'].max())
        ax.set_ylim(0, max_value * 1.2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        
        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Customer {cid}" for cid in plot_data['CustomerID']])
        
        ax.set_title('Customer CLV vs Dynamic Pricing', fontsize=16, pad=20, fontweight='bold')
        ax.set_ylabel('Value (USD)', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add legend and adjust layout
        ax.legend(title="Metrics", labels=['CLV', 'Dynamic Price'], 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save and show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nChart saved to: {os.path.abspath(save_path)}")
        plt.show()

# Example usage
if __name__ == "__main__":
    try:
        # Initialize with your paths (using raw strings for Windows)
        visualizer = CLVVisualizer(
            model_path=r"C:\Users\SherAsghar\Desktop\DPE\models\clv_model.pkl",
            results_path=r"C:\Users\SherAsghar\Desktop\DPE\results\clv_results.csv"
        )
        
        # Option 1: Plot random samples
        print("\nVisualizing random samples...")
        visualizer.plot_clv_vs_price()
        
        # Option 2: Plot specific customers (uncomment and replace with real IDs)
        # print("\nVisualizing specific customers...")
        # visualizer.plot_clv_vs_price(customer_ids=['CUST001', 'CUST123', 'CUST456'])
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nDebugging Tips:")
        print("1. Verify the model and CSV files exist at the specified paths")
        print("2. Check the CSV contains 'CustomerID' and 'Predicted_CLV' columns")
        print("3. Ensure the CLV values are reasonable (not all zeros)")