import os
from dotenv import load_dotenv

load_dotenv()

OPENF1_BASE_URL = "https://api.openf1.org/v1"
DB_PATH = "data/f1_assistant.db"
DATASET_PATH = "data/f1_dataset_v2.csv"
MODEL_SAVE_PATH = "models/"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
