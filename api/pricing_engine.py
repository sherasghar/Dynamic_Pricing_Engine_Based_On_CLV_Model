import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class PricingEngine:
    def __init__(self, model_path: str, base_price: float = 100.0):
        self.model_path = model_path
        self.base_price = base_price
        self.model = self._load_model()
        
    def _load_model(self) -> BaseEstimator:
        try:
            model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def calculate_clv(self, customer_data: dict) -> float:
        try:
            input_df = pd.DataFrame([customer_data])
            required_features = ['Recency', 'Frequency', 'MonetaryValue', 'Tenure',
                               'AvgDaysBetweenPurchases', 'Age', 'UniqueProductsCount']
            missing_features = set(required_features) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
                
            clv = self.model.predict(input_df[required_features])[0]
            return max(0, clv)
        except Exception as e:
            logger.error(f"CLV calculation failed: {str(e)}")
            raise RuntimeError(f"CLV calculation error: {str(e)}")
    
    def calculate_dynamic_price(self, customer_data: dict, product_cost: float = 50.0) -> dict:
        try:
            clv = self.calculate_clv(customer_data)
            clv_factor = self._normalize_clv(clv)
            dynamic_price = max(product_cost * 1.1, self.base_price * clv_factor)
            
            return {
                "base_price": self.base_price,
                "dynamic_price": round(dynamic_price, 2),
                "clv": round(clv, 2),
                "price_adjustment_factor": round(clv_factor, 2),
                "min_price": round(product_cost * 1.1, 2),
                "profit_margin": round((dynamic_price - product_cost) / dynamic_price * 100, 2)
            }
        except Exception as e:
            logger.error(f"Price calculation failed: {str(e)}")
            raise RuntimeError(f"Price calculation error: {str(e)}")
    
    def _normalize_clv(self, clv: float) -> float:
        low_clv = 100
        high_clv = 1000
        normalized = np.clip((clv - low_clv) / (high_clv - low_clv), 0, 1)
        return 0.8 + (0.4 * normalized)