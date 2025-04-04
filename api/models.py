from pydantic import BaseModel
from typing import List

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