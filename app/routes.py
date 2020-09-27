from app import app
from flask import render_template, send_from_directory
import pandas as pd


@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)

@app.route("/", methods=['GET'])
def home():
    # This is where we pass in restaurants
    restaurants = pd.read_csv('app/data/Restaurant_links.csv')
    restaurants = restaurants['name'] + [", " for i in range(restaurants.shape[0])] + restaurants['zip_code']
    restaurants = restaurants.sort_values()
    return render_template('home.html', restaurants=restaurants)

@app.route("/results") 
def results():
    #Do something here
    return render_template('result.html')