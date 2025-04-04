import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
df = pd.read_excel('C:/Users/SherAsghar/Desktop/dynamic_pricing_engine/data/raw/online_retail.xlsx')

# 1. Data Cleaning
# Remove rows with missing CustomerID (these can't be used for CLV)
df = df[df['CustomerID'].notna()]

# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)

# Handle negative quantities (returns) - we'll keep them for accurate revenue calculation
# Calculate revenue for each transaction
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 2. Feature Engineering
# Extract date features
df['InvoiceYearMonth'] = df['InvoiceDate'].dt.to_period('M')
df['InvoiceDay'] = df['InvoiceDate'].dt.day
df['InvoiceDayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['InvoiceHour'] = df['InvoiceDate'].dt.hour

# 3. Create CLV Dataset - Customer Level Aggregation
# Set snapshot date (last date in dataset + 1 day)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Create historical data for CLV
clv_data = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'Revenue': 'sum'                                         # Monetary Value
}).reset_index()

# Rename columns
clv_data.columns = ['CustomerID', 'Recency', 'Frequency', 'MonetaryValue']

# Calculate average monetary value per transaction
clv_data['MonetaryValue'] = clv_data['MonetaryValue'] / clv_data['Frequency']

# 4. Additional Customer Features
# Customer tenure (days since first purchase)
customer_tenure = df.groupby('CustomerID')['InvoiceDate'].agg(['min', 'max'])
customer_tenure['Tenure'] = (customer_tenure['max'] - customer_tenure['min']).dt.days
clv_data = clv_data.merge(customer_tenure[['Tenure']], on='CustomerID', how='left')

# Average days between purchases
clv_data['AvgDaysBetweenPurchases'] = clv_data['Tenure'] / clv_data['Frequency']

# 5. Demographic Features (if available)
# Check if demographic columns exist before processing
demo_features_list = []
if 'Age' in df.columns:
    demo_features_list.append(('Age', 'mean'))
if 'Gender' in df.columns:
    demo_features_list.append(('Gender', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'))
if 'Country' in df.columns:
    demo_features_list.append(('Country', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'))

if demo_features_list:
    demo_features = df.groupby('CustomerID').agg(dict(demo_features_list)).reset_index()
    clv_data = clv_data.merge(demo_features, on='CustomerID', how='left')

# 6. Product Preferences
# Count of unique products purchased
unique_products = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
unique_products.columns = ['CustomerID', 'UniqueProductsCount']
clv_data = clv_data.merge(unique_products, on='CustomerID', how='left')

# 7. Save preprocessed data
clv_data.to_csv('C:/Users/SherAsghar/Desktop/dynamic_pricing_engine/data/processed/clv_preprocessed_data.csv', index=False)

print("Preprocessing complete. Data saved to clv_preprocessed_data.csv")
print(f"Final dataset shape: {clv_data.shape}")
print(clv_data.head())