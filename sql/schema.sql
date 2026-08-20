-- PostgreSQL Schema for Bank Customer Churn Prediction System

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    surname VARCHAR(255),
    credit_score INTEGER,
    geography VARCHAR(100),
    gender VARCHAR(50),
    age INTEGER,
    tenure INTEGER,
    balance DOUBLE PRECISION,
    num_of_products INTEGER,
    has_cr_card INTEGER,
    is_active_member INTEGER,
    estimated_salary DOUBLE PRECISION,
    exited INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers(customer_id);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    model_version VARCHAR(100),
    churn_probability DOUBLE PRECISION,
    predicted_churn BOOLEAN,
    risk_level VARCHAR(50),
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_customer ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_pred_churn ON predictions(predicted_churn);
CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_pred_model_version ON predictions(model_version);

CREATE TABLE IF NOT EXISTS model_metadata (
    model_version VARCHAR(100) PRIMARY KEY,
    model_type VARCHAR(100),
    training_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    training_rows INTEGER,
    recall DOUBLE PRECISION,
    precision DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    roc_auc DOUBLE PRECISION,
    pr_auc DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    artifact_path VARCHAR(255)
);

-- Why these indexes?
-- idx_customers_customer_id: Fast lookup of customer details when a prediction request comes in.
-- idx_pred_customer: Allows quick retrieval of all historical predictions for a specific customer.
-- idx_pred_churn: Useful for quickly aggregating how many customers are predicted to churn (e.g. for monitoring dashboards).
-- idx_pred_timestamp: Crucial for time-series monitoring, detecting drift, and computing daily API usage.
