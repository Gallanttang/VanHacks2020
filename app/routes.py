from app import app
from flask import render_template, send_from_directory


@app.route("/static/<path:path>")
def static_dir(path):
    return send_from_directory("static", path)


@app.route("/")
def home():
    # Do something here
    return render_template('home.html')


@app.route("/results")
def results():
    # Do something here
    return render_template('result.html')