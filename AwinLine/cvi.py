########### Python 2.7 #############
import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
import httplib
import urllib
import base64
import json
###############################################
#### Update or verify the following values. ###
###############################################

# Replace the subscription_key string value with your valid subscription key.
subscription_key = '1673956521414055bf1337e6bc5769f9'

# Replace or verify the region.
#
# You must use the same region in your REST API call as you used to obtain your subscription keys.
# For example, if you obtained your subscription keys from the westus region, replace
# "westcentralus" in the URI below with "westus".
#
# NOTE: Free trial subscription keys are generated in the westcentralus region, so if you are using
# a free trial subscription key, you should not need to change this region.
uri_base = 'westcentralus.api.cognitive.microsoft.com'

headers = {
    # Request headers.
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

params = urllib.urlencode({
    # Request parameters. All of them are optional.
    'visualFeatures': 'Categories,Description,Color',
    'language': 'en',
})

# The URL of a JPEG image to analyze.
body = "{'url':'https://9f50b2e1.ngrok.io/wb/6774503903607.jpg'}"

try:
    # Execute the REST API call and get the response.
    conn = httplib.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/vision/v1.0/analyze?%s" %
                 params, open(os.path.join(fileDir, "wb") + '/6774503903607.jpg', 'rb'), headers)
    response = conn.getresponse()
    data = response.read()

    # 'data' contains the JSON data. The following formats the JSON data for display.
    parsed = json.loads(data)
    print("Response:")
    #print(json.dumps(parsed, sort_keys=True, indent=2))
    conn.close()
    print(parsed.get("description").get("captions")[0].get("text"))

except Exception as e:
    print('Error:')
    print(e)

####################################
