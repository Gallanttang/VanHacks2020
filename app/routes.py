from app import app

@app.route("/")
def home():
    return "Hello, World!"
    
if __init__ == "__main__":
    app.run(debug=True)
