from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import os
import joblib
import pandas as pd
import numpy as np
import logging
from pydantic import BaseModel
from typing import List

# Initialize app
app = FastAPI(title="Dynamic Pricing Engine")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Configure static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Configure templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Pydantic models
class CustomerData(BaseModel):
    Recency: float
    Frequency: float
    MonetaryValue: float
    Tenure: float
    AvgDaysBetweenPurchases: float
    Age: float
    UniqueProductsCount: float
    product_cost: float = 50.0

class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

# Pricing Engine Class
class PricingEngine:
    def __init__(self, model_path: str, base_price: float = 100.0):
        self.model_path = model_path
        self.base_price = base_price
        self.model = self._load_model()
        
    def _load_model(self):
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

# Initialize pricing engine
try:
    model_path = str(BASE_DIR / "models/clv_model.pkl")
    pricing_engine = PricingEngine(model_path=model_path, base_price=100.0)
    logger.info("Pricing engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pricing engine: {str(e)}")
    raise RuntimeError("Could not start application - pricing engine failed")

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/calculate_price/")
async def calculate_price(customer: CustomerData):
    try:
        result = pricing_engine.calculate_dynamic_price(
            customer_data=customer.dict(),
            product_cost=customer.product_cost
        )
        return JSONResponse({"status": "success", "data": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/calculate_batch_prices/")
async def calculate_batch_prices(batch_data: BatchCustomerData):
    try:
        results = []
        for customer in batch_data.customers:
            result = pricing_engine.calculate_dynamic_price(
                customer_data=customer.dict(),
                product_cost=customer.product_cost
            )
            results.append(result)
        return JSONResponse({"status": "success", "data": results})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/test_model/")
async def test_model():
    try:
        test_customer = {
            "Recency": 30,
            "Frequency": 5,
            "MonetaryValue": 500,
            "Tenure": 365,
            "AvgDaysBetweenPurchases": 30,
            "Age": 35,
            "UniqueProductsCount": 3,
            "product_cost": 50.0
        }
        
        result = pricing_engine.calculate_dynamic_price(test_customer)
        
        if not isinstance(result["dynamic_price"], float):
            raise ValueError("Invalid price prediction")
        if result["dynamic_price"] < result["min_price"]:
            raise ValueError("Price below minimum threshold")
            
        return JSONResponse({
            "status": "success",
            "message": "Model working correctly",
            "test_result": result,
            "test_input": test_customer
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model test failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Dynamic Pricing Engine is running"}

# Error handler
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Endpoint not found"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)