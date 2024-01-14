import pandas as pd
from flask import Flask, jsonify, render_template, request
from zbiorniczek import slipperyZbiorniczek

app = Flask(__name__)
slippery = slipperyZbiorniczek()

def run_optimisation(method="Topsis"):
    df = slippery.run_algorithm(method)
    df_ = df.reset_index(drop=True)
    return df_


@app.route("/results/UTA", methods=("POST", "GET"))
def results_UTA():
    df = run_optimisation("UTA")
    return render_template("results.html", table_html=[df.to_html(classes='data', header='true')], method="UTA")

@app.route("/results/topsis", methods=("POST", "GET"))
def results_topsis():
    df = run_optimisation("Topsis")
    return render_template("results.html", table_html=[df.to_html(classes='data', header='true')], method="Topsis")

@app.route("/results/fuzzy", methods=("POST", "GET"))
def results_fuzzy():
    df = run_optimisation("Fuzzy")
    return render_template("results.html", table_html=[df.to_html(classes='data', header='true')], method="Fuzzy Topsis")

@app.route("/show_data", methods=("POST", "GET"))
def show_data():
    data_path = request.files['file'].filename
    try:
        slippery.load_data_from_file(data_path)
        df = slippery.loaded_data.copy()
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