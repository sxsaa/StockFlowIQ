import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Construct the DATABASE_URL
DATABASE_URL = f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"

print(DATABASE_URL)  # Optional: Print to verify the URL