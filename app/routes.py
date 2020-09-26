from app import app
from flask import render_template

@app.route("/")
def home():
    # Do something here
    return render_template('home.html')

