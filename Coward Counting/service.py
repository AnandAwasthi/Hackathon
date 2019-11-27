import requests
import json

url = 'http://localhost:5000/api/'

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
image_path = 'D://Personal//Hackathon//Coward Counting//test//IMG_3.jpg'
r = requests.get(url, data = image_path, headers=headers)
print(r, r.text)