from app.services.supabase_client import supabase
from datetime import datetime

def save_meeting_to_db(mom: dict):
    data = {
        "meeting_topic": mom.get("meeting_topic"),
        "meeting_date": datetime.utcnow().isoformat(),
        "meeting_summary": mom.get("meeting_summary"),
        "action_items": mom.get("action_items"),
        "open_points": mom.get("open_points"),
        "mom": mom
    }

    print("Before insert")

    response = supabase.table("meetings_held").insert(data).execute()

    print("After insert")
   
   

    if response.data is None:
        print("Supabase error")
        raise Exception(response.error)
    
    print("Supabase data:", response.data)    
    return response.data[0]
