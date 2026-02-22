import os
from dotenv import load_dotenv

load_dotenv()





GROQ_API_KEY=os.getenv("GROQ_API_KEY")
Hugging_Face_Hub_Tokken=os.getenv("Hugging_Face_Hub_Tokken")

MODEL="llama-3.3-70b-versatile"
