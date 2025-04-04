import os
import joblib
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLVModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self._validate_paths()
        
    def _validate_paths(self):
        """Ensure all directories in paths exist"""
        for path in [self.config['model_path'], self.config['results_path']]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Verified directory exists: {os.path.dirname(path)}")

    def load_data(self):
        """Load and prepare data"""
        try:
            logger.info(f"Loading data from {self.config['data_path']}")
            self.df = pd.read_csv(self.config['data_path'])
            
            # Validate required columns exist
            required_columns = self.config['features'] + [self.config['target']]
            missing_cols = set(required_columns) - set(self.df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in data: {missing_cols}")
                
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def train_model(self):
        """Train the Random Forest model"""
        try:
            X = self.df[self.config['features']]
            y = self.df[self.config['target']]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['test_size'], 
                random_state=self.config['random_state']
            )

            # Initialize and train model
            self.model = RandomForestRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state']
            )
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5  # Calculate RMSE manually
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model trained successfully - RMSE: {rmse:.2f}, R²: {r2:.2f}")
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def save_results(self):
        """Save model and predictions"""
        try:
            # Save model
            joblib.dump(self.model, self.config['model_path'])
            logger.info(f"Model saved to {self.config['model_path']}")

            # Save predictions
            self.df['Predicted_CLV'] = self.model.predict(self.df[self.config['features']])
            self.df[['CustomerID', 'Predicted_CLV']].to_csv(self.config['results_path'], index=False)
            logger.info(f"Predictions saved to {self.config['results_path']}")
            return True
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

# Configuration
CONFIG = {
    'data_path': "C:/Users/SherAsghar/Desktop/DYNAMIC_PRICING_ENGINE/data/processed/clv_preprocessed_data.csv",
    'model_path': "C:/Users/SherAsghar/Desktop/DYNAMIC_PRICING_ENGINE/models/clv_model.pkl",
    'results_path': "C:/Users/SherAsghar/Desktop/DYNAMIC_PRICING_ENGINE/results/clv_results.csv",
    'features': ['Recency', 'Frequency', 'MonetaryValue', 'Tenure', 
                'AvgDaysBetweenPurchases', 'Age', 'UniqueProductsCount'],
    'target': 'MonetaryValue',
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 200,
    'max_depth': 10
}

if __name__ == "__main__":
    trainer = CLVModelTrainer(CONFIG)
    
    if (trainer.load_data() and 
        trainer.train_model() and 
        trainer.save_results()):
        logger.info("✅ CLV pipeline completed successfully!")
    else:
        logger.error("❌ CLV pipeline failed")