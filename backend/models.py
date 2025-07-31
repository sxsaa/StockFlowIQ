from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Feedback(Base):
    __tablename__ = "cust_reviews"
    
    customer_id = Column(Integer, primary_key=True, index=True)
    feedback = Column(String, nullable=False)
    status = Column(Integer, nullable=False)  # 0: Neutral, 1: Negative, 2: Positive
