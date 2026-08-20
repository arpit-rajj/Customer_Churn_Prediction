from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    customer_id: int = Field(..., description="Unique identifier for the customer")
    credit_score: int = Field(..., ge=300, le=850, description="Customer's credit score (300-850)")
    geography: str = Field(..., description="Customer's country")
    gender: str = Field(..., description="Customer's gender (Male/Female)")
    age: int = Field(..., ge=18, le=100, description="Customer's age")
    tenure: int = Field(..., ge=0, le=10, description="Number of years the customer has been with the bank")
    balance: float = Field(..., ge=0, description="Account balance")
    num_of_products: int = Field(..., ge=1, le=4, description="Number of bank products the customer uses")
    has_cr_card: int = Field(..., ge=0, le=1, description="Whether the customer has a credit card (1=Yes, 0=No)")
    is_active_member: int = Field(..., ge=0, le=1, description="Whether the customer is an active member (1=Yes, 0=No)")
    estimated_salary: float = Field(..., ge=0, description="Estimated salary of the customer")

class PredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    predicted_churn: bool
    risk_level: str
    model_version: str
