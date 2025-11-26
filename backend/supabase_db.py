from supabase import create_client
from backend.config import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_aqi_record(record):
    return supabase.table("aq_data").insert(record).execute()

def get_city_data(city):
    return supabase.table("aq_data").select("*").eq("city", city).execute()
