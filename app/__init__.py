from flask import Flask

# app will be the object that hosts the web application
# templates is the directory that is holds all of the HTML files relevant to the game
# static_folder holds all the css files (Not used)
app = Flask(__name__, template_folder='templates', static_folder='static')

