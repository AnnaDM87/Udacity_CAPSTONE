import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "Inputs": {
    "data": [
      {
        "Column2": "example_value",
        "JoiningYear": 0,
        "PaymentTier": 0,
        "Age": 0,
        "ExperienceInCurrentDomain": 0,
        "Education_Masters": 0,
        "Education_PHD": 0,
        "Gender_Male": 0,
        "City_New Delhi": 0,
        "City_Pune": 0,
        "EverBenched_Yes": 0
      }
    ]
  },
  "GlobalParameters": {
    "method": "predict"
  }
}

body = str.encode(json.dumps(data))

url = 'http://5be99864-5c61-4bef-8e54-dff484451595.westeurope.azurecontainer.io/score'
# Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = ''
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")


headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))