import pandas as pd
from flask import Flask, jsonify, render_template
import numpy as np

app = Flask(__name__)

@app.route("/data", methods=("POST", "GET"))
def load_data():

    # TODO: import danych i wybór kryteriów
    # df = load_dataframe()
    df = pd.DataFrame()
    return render_template("data.html", table_html=[df.to_html(classes='data', header='true')])

@app.route("/", methods=("POST", "GET"))
def home_screen():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)