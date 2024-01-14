import pandas as pd
from flask import Flask, jsonify, render_template, request
import numpy as np
from tkinter import filedialog, Tk

import time

app = Flask(__name__)
def run_optimisation(method="TOPSIS"):
    if (method == "TOPSIS"):
        # df = get_topsis_results()
        df = pd.DataFrame()
    
    return df

@app.route("/results", methods=("POST", "GET"))
def results():
    df = run_optimisation()
    return render_template("results.html", table_html=[df.to_html(classes='data', header='true', index=False)])

@app.route("/show_data", methods=("POST", "GET"))
def show_data():
    f = request.files['file']
    try:
        df = pd.read_csv(f)
        df.columns.values[0] = 'ID'
    except:
        df = pd.DataFrame()
    return render_template("show_data.html", table_html=[df.to_html(classes='data', header='true', index=False)])

@app.route("/load_data", methods=("POST", "GET"))
def load_data():
    return render_template("load_data.html")

@app.route("/", methods=("POST", "GET"))
def home_screen():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)