# save mom.json in system , should be temporarily
import json
import os

MOM_OUTPUT_PATH = "app/Working_transcript/mom.json"

def save_mom(mom_data: dict) -> None:
    os.makedirs(os.path.dirname(MOM_OUTPUT_PATH), exist_ok=True)

    with open(MOM_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mom_data, f, indent=2, ensure_ascii=False)
