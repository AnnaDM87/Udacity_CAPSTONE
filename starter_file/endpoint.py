import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://df676f91-6bcf-4a77-aa03-567a40949083.westus2.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'lIaXZhVbMKl6FkfSkJjMzt44GWlAyXPT'

# Two sets of data to score, so we get two results back
data =  {
  "Inputs": {
    "data": [
      {
        "Column2": "example_value",
        "JoiningYear": 2018,
        "PaymentTier": 3,
        "Age": 25,
        "ExperienceInCurrentDomain": 3,
        "Education_Masters": 1,
        "Education_PHD": 0,
        "Gender_Male": 1,
        "City_New Delhi": 0,
        "City_Pune": 0,
        "EverBenched_Yes": 1
      },
      {
        "Column2": "example_value",
        "JoiningYear": 2015,
        "PaymentTier": 3,
        "Age": 22,
        "ExperienceInCurrentDomain": 0,
        "Education_Masters": 0,
        "Education_PHD": 0,
        "Gender_Male": 1,
        "City_New Delhi": 1,
        "City_Pune": 0,
        "EverBenched_Yes": 0
      }
    ]
  },
  "GlobalParameters": {
    "method": "predict"
  }
}

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


