from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="API to predict customer churn based on a trained Random Forest model.",
    version="1.0.0"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
