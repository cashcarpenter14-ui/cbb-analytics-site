import os
import pandas as pd
import requests

# Paths
CSV_PATH = "data/team_branding.csv"
SAVE_DIR = "assets/team_logos"

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

success = 0
fail = 0

for _, row in df.iterrows():
    team = str(row.get("team", "")).strip()
    url = str(row.get("logo_url", "")).strip()
    file_name = str(row.get("logo_file", "")).strip()

    if not url.startswith("http") or file_name == "" or file_name == "nan":
        fail += 1
        continue

    save_path = os.path.join(SAVE_DIR, file_name)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            success += 1
            print(f"Downloaded: {team}")
        else:
            fail += 1
            print(f"Failed (status): {team}")
    except Exception as e:
        fail += 1
        print(f"Failed (error): {team}")

print(f"\nDone. Success: {success}, Failed: {fail}")