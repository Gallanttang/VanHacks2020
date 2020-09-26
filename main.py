from bs4 import BeautifulSoup
from selenium import webdriver
import time
import numpy as np
import requests
import json
import csv

# restaurant_name = []
# restaurant_add = []
# restaurant_cuisine = []
# restaurant_ranking = []
# restaurant_rating = []
# serving_time = []
# price_range = []
# review_num = []
# visit_year = []
# visit_month = []
# review_year = []
# review_month = []
# review_day = []
# overall_rating = []
# value = []
# service = []
# food = []
# review_title = []
# review_content = []
offset = 50
chrome_driver_path = '/usr/local/bin/chromedriver'
lat = []
long = []
headers = {
    'Authorization': 'Bearer [put API auth key here]',
    'content-type': 'applications/json',
    'ratelimit-dailylimit': '5000'
}

for j in range(-5, 5):
    long.append(float(-122.4194 + float(j/200)))
    lat.append(float(37.7749 + float(j/200)))
long = np.unique(np.array(long))
lat = np.unique(np.array(lat))

with open('data/Restaurant_links.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    results = {}
    counter = 0
    for j in long:
        for h in lat:
            for i in range(20):
                if counter > 4999:
                    break;
                counter += 1
                try:
                    offset = 50 * i
                    link = "https://api.yelp.com/v3/businesses/search?latitude=" \
                           + str(h) + "&longitude=" \
                           + str(j) + "&radius=1000" \
                           + "&limit=50&offset=" + str(offset)
                    print(link)
                    r = requests.get(link, headers=headers)
                    json_loaded = json.loads(r.text)
                    if json_loaded['businesses'] is None or len(json_loaded['businesses']) < 1:
                        break
                    for business in json_loaded['businesses']:
                        csv_writer.writerow([business['id'], business['name']])
                except:
                    continue
