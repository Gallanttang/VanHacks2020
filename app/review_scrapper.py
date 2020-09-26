import json
import html
import pandas as pd
import requests
import json
import csv
import html
import time

def transform_link(x):
    return str(x) + "/review_feed"

# chrome_driver_path = '/usr/local/bin/chromedriver'

df = pd.read_csv('data/Restaurant_links.csv', index_col='id')

df['review_url'] = df['url'].apply(lambda x: x[:x.find('biz/') + 4]) + df.index 
df['review_url'] = df['review_url'].apply(transform_link)

with open('data/vancouver_reviews.csv', "w") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter='\t')
    csv_writer.writerow(['business_id', 'user_id', 'rating', 'comment', 'feedback', "time_uploaded"])
    tracking = 0
    for row in df['review_url']: 
        if tracking > 2:
            break
        tracking += 1 
        print(row)
        ENDPOINT = row
        PARAMETERS = {
            'rl': 'en',
            'sort_by': "date_desc",
            'start': 0
        }
        keep_searching = True
        counter = 1
        while keep_searching:
            if counter > 50:
                break
            r = requests.get(url = ENDPOINT, params = PARAMETERS)
            json_loaded = json.loads(r.text)
            reviews = json_loaded['reviews']
            if len(reviews) < 1:
                keep_searching = False
            else:
                PARAMETERS['start'] += 20
                for review in reviews:
                    comment = html.unescape(review['comment']['text'])
                    csv_writer.writerow([review['business']['id'], review['userId'], review['rating'], comment, review['feedback']['counts'], review['localizedDate']])
                    counter += 1
        time.sleep(5)


