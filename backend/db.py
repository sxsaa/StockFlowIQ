from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

# Create database engine
engine = create_engine(DATABASE_URL)

# Create a session
SessionLocal = sessionmaker(bind=engine)
