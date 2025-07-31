# from fastapi import FastAPI, Depends
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from transformers import pipeline
# from db import SessionLocal, engine
# from models import Base, Feedback

# app = FastAPI()
# Base.metadata.create_all(bind=engine)

# # Load RoBERTa model
# sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

# # Define Pydantic model for request body
# class FeedbackRequest(BaseModel):
#     customer_id: int  # Ensure customer_id is included
#     feedback_text: str

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @app.post("/analyze/")
# def analyze_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
#     """
#     Analyze user feedback using RoBERTa and store sentiment in the database.
#     """
#     # Run RoBERTa model
#     result = sentiment_pipeline(request.feedback_text)[0]
#     sentiment_label = result["label"]

#     # Debugging: Print sentiment result
#     print(f"Feedback: {request.feedback_text}, Sentiment: {sentiment_label}")

#     # Correct label mapping
#     status_map = {"LABEL_0": 1, "LABEL_1": 0, "LABEL_2": 2}  # Negative = 1, Neutral = 0, Positive = 2
#     sentiment_status = status_map.get(sentiment_label, 0)  # Default Neutral (0)

#     # Save in DB only if feedback is negative
#     if sentiment_status == 1:
#         feedback_entry = Feedback(customer_id=request.customer_id, feedback=request.feedback_text, status=sentiment_status)
#         db.add(feedback_entry)
#         db.commit()
#         db.refresh(feedback_entry)

#     return {"feedback": request.feedback_text, "sentiment": sentiment_label, "status": sentiment_status}


# from fastapi import FastAPI, Depends
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from transformers import pipeline
# import pandas as pd
# import statsmodels.api as sm

# from db import SessionLocal, engine
# from models import Base, Feedback

# app = FastAPI()
# Base.metadata.create_all(bind=engine)

# # Load RoBERTa model for sentiment analysis
# sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

# # Define Pydantic model for request body
# class FeedbackRequest(BaseModel):
#     customer_id: int  # Ensure customer_id is included
#     feedback_text: str

# # Database Dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @app.post("/analyze/")
# def analyze_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
#     """
#     Analyze user feedback using RoBERTa and store sentiment in the database.
#     """
#     # Run RoBERTa model
#     result = sentiment_pipeline(request.feedback_text)[0]
#     sentiment_label = result["label"]

#     # Debugging: Print sentiment result
#     print(f"Feedback: {request.feedback_text}, Sentiment: {sentiment_label}")

#     # Correct label mapping
#     status_map = {"LABEL_0": 1, "LABEL_1": 0, "LABEL_2": 2}  # Negative = 1, Neutral = 0, Positive = 2
#     sentiment_status = status_map.get(sentiment_label, 0)  # Default Neutral (0)

#     # Save in DB only if feedback is negative
#     if sentiment_status == 1:
#         feedback_entry = Feedback(customer_id=request.customer_id, feedback=request.feedback_text, status=sentiment_status)
#         db.add(feedback_entry)
#         db.commit()
#         db.refresh(feedback_entry)

#     return {"feedback": request.feedback_text, "sentiment": sentiment_label, "status": sentiment_status}

# # --- SALES FORECASTING API ---

# # Load dataset
# store_sales = pd.read_csv("C:\\Users\\Kavindu Sasanka\\Documents\\Academics\\University\\Semester 6\\Data Management Project\\Objective Models\\Sales Forecasting\\ARIMA Model\\SeriesReport-Not Seasonally Adjusted Sales - Monthly (Millions of Dollars).csv")
# store_sales.dropna(inplace=True)

# # Convert 'Period' column to datetime
# store_sales['Period'] = pd.to_datetime(store_sales['Period'])

# # Define and fit SARIMA model
# model = sm.tsa.statespace.SARIMAX(store_sales['Value'], order=(0, 1, 1), seasonal_order=(1, 1, 1, 12))
# results = model.fit()

# # Generate predictions
# store_sales['forecast'] = results.predict(start=327, end=340, dynamic=True)

# @app.get("/forecast/")
# def get_forecast():
#     """
#     API endpoint to return sales forecast data.
#     """
#     forecast_data = [["Year", "Sales Forecast"]]  # Google Charts format
#     for index, row in store_sales[['Period', 'forecast']].dropna().iterrows():
#         forecast_data.append([row["Period"].strftime("%Y-%m"), row["forecast"]])  # Convert datetime to string

#     return JSONResponse(content=forecast_data)

from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from transformers import pipeline
import pandas as pd
import statsmodels.api as sm
from db import SessionLocal, engine
from models import Base, Feedback

app = FastAPI()

# Middleware to allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# Load RoBERTa model for sentiment analysis
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

# Define Pydantic model for request body
class FeedbackRequest(BaseModel):
    customer_id: int  # Ensure customer_id is included
    feedback_text: str

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/analyze/")
def analyze_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Analyze user feedback using RoBERTa and store sentiment in the database.
    """
    # Run RoBERTa model
    result = sentiment_pipeline(request.feedback_text)[0]
    sentiment_label = result["label"]

    # Debugging: Print sentiment result
    print(f"Feedback: {request.feedback_text}, Sentiment: {sentiment_label}")

    # Correct label mapping
    status_map = {"LABEL_0": 1, "LABEL_1": 0, "LABEL_2": 2}  # Negative = 1, Neutral = 0, Positive = 2
    sentiment_status = status_map.get(sentiment_label, 0)  # Default Neutral (0)

    # Save in DB only if feedback is negative
    if sentiment_status == 1:
        feedback_entry = Feedback(customer_id=request.customer_id, feedback=request.feedback_text, status=sentiment_status)
        db.add(feedback_entry)
        db.commit()
        db.refresh(feedback_entry)

    return {"feedback": request.feedback_text, "sentiment": sentiment_label, "status": sentiment_status}

# --- SALES FORECASTING API --- #

# Load dataset
store_sales = pd.read_csv("C:\\Users\\Kavindu Sasanka\\Documents\\Academics\\University\\Semester 6\\Data Management Project\\Objective Models\\Sales Forecasting\\ARIMA Model\\SeriesReport-Not Seasonally Adjusted Sales - Monthly (Millions of Dollars).csv")  # Update with actual path
store_sales.dropna(inplace=True)

# Convert 'Period' column to datetime
store_sales['Period'] = pd.to_datetime(store_sales['Period'])

# Define and fit SARIMA model
model = sm.tsa.statespace.SARIMAX(store_sales['Value'], order=(0, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Generate predictions
store_sales['forecast'] = results.predict(start=327, end=340, dynamic=True)

@app.get("/forecast/")
def get_forecast():
    """
    API endpoint to return sales forecast data.
    """
    forecast_data = [["Year", "Sales Forecast"]]  # Google Charts format
    for index, row in store_sales[['Period', 'forecast']].dropna().iterrows():
        forecast_data.append([row["Period"].strftime("%Y-%m"), row["forecast"]])  # Convert datetime to string

    return JSONResponse(content=forecast_data)


