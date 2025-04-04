import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import numpy as np

# Load your trained model
model = joblib.load('models/clv_model.pkl')

# You'll need your feature names - replace with your actual feature names
feature_names = ['Recency', 'Frequency', 'MonetaryValue', 'Tenure', 
                'AvgDaysBetweenPurchases', 'Age', 'UniqueProductsCount']

# Extract feature importances
importances = model.feature_importances_

# Create and sort importance dataframe
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Create the visualization
plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Importance', y='Feature', data=importance_df,
                palette='coolwarm', edgecolor='black', linewidth=1)

# Add value labels
for i, (feature, imp) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
    ax.text(imp + 0.005, i, f'{imp:.3f}', 
            va='center', fontsize=11, color='black')

# Customize the plot
plt.title('Customer Lifetime Value - Feature Importance', 
         fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Relative Importance', fontsize=13)
plt.ylabel('')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# Add reference line at average importance
avg_importance = importances.mean()
plt.axvline(avg_importance, color='gray', linestyle='--', alpha=0.7)
plt.text(avg_importance + 0.01, len(feature_names)/2, 
        f' Average\n({avg_importance:.3f})',
        va='center', fontsize=10, color='gray')

# Improve layout
sns.despine(left=True, bottom=True)
plt.tight_layout()

# Save and show
plt.savefig('clv_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()