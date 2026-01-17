import requests
import json

def test_api():
    url = "http://localhost:8000/ai-move"
    
    # Mock game state
    payload = {
        "dice": [1, 2, 3, 4, 5], # Large Straight
        "rolls_remaining": 2,
        "scorecard": {
            "ones": None,
            "chance": None
        },
        "difficulty": "neural"
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("SUCCESS: API returned 200 OK")
            print("Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"ERROR: API returned {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"ERROR: Connection failed: {e}")

if __name__ == "__main__":
    test_api()
