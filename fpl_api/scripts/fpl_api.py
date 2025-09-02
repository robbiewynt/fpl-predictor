import requests
import pandas as pd

# Download general bootstrap data
def get_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    return data

# Save player stats to csv
def save_player_data():
    data = get_bootstrap()
    players = pd.DataFrame(data['elements'])
    players.to_csv("data/players.csv", index=False)

if __name__ == "__main__":
    save_player_data()