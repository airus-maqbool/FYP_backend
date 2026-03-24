import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
print(os.getenv("SUPABASE_URL"))
print(os.getenv("SUPABASE_SERVICE_KEY"))