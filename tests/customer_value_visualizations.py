import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Generate sample CLV data (replace with your actual data)
np.random.seed(42)
clv_data = pd.DataFrame({
    'CLV': np.random.gamma(2, 500, 1000),
    'Recency': np.random.randint(1, 365, 1000),
    'Frequency': np.random.poisson(5, 1000),
    'MonetaryValue': np.random.normal(500, 200, 1000).clip(10),
    'Tenure': np.random.randint(30, 365*3, 1000),
    'AvgDaysBetweenPurchases': np.random.randint(1, 90, 1000),
    'Age': np.random.randint(18, 80, 1000),
    'UniqueProductsCount': np.random.randint(1, 15, 1000)
})

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# 1. CLV Distribution Plot
plt.figure(figsize=(12, 6))
ax = sns.histplot(data=clv_data, x='CLV', bins=30, 
                 kde=True, color='royalblue',
                 edgecolor='white', linewidth=0.5,
                 alpha=0.8)

# Add annotations
mean_clv = clv_data['CLV'].mean()
median_clv = clv_data['CLV'].median()
plt.axvline(mean_clv, color='red', linestyle='--', linewidth=1.5)
plt.axvline(median_clv, color='green', linestyle='--', linewidth=1.5)

plt.text(mean_clv*1.1, plt.ylim()[1]*0.9, 
         f'Mean: ${mean_clv:,.0f}', 
         color='red', fontsize=12)
plt.text(median_clv*0.7, plt.ylim()[1]*0.8, 
         f'Median: ${median_clv:,.0f}', 
         color='green', fontsize=12)

# Customize
plt.title('Customer Lifetime Value Distribution', fontsize=16, pad=20)
plt.xlabel('CLV (USD)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig('clv_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 8))

# Calculate correlations
corr = clv_data.corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Plot heatmap
ax = sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap='coolwarm', center=0,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})

# Customize
plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('clv_correlation.png', dpi=300, bbox_inches='tight')
plt.show()