# %% Establishing connection to GDELT API and fetching data
import requests

url = "https://api.gdeltproject.org/api/v2/doc/doc"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
params = {"query": '("islamic state" OR isis OR daesh)',
          "mode": "artlist",
          "maxrecords": 10,
          "timespan": "1week",
          "format": "json"}
resp = requests.get(url, params=params, headers=headers)
print("Status Code:", resp.status_code)
data = resp.json()
print(data.keys())
# %% Exploring the structure of the returned data
data["articles"][0].keys()
data["articles"][0]