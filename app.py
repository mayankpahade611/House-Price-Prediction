from flask import Flask, request, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model and columns
model = pickle.load(open("banglore_home_prices_model.pickle", "rb"))
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        sqft = float(request.form['sqft'])
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])
        location = request.form['location'].lower()

        # Create input array with all zeros
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        # If location exists in data_columns, set 1
        if location in data_columns:
            loc_index = data_columns.index(location)
            x[loc_index] = 1

        # Predict
        predicted_price = round(model.predict([x])[0], 2)

        return render_template("index.html", prediction_text=f"Estimated Price: {predicted_price} Lakhs")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render provides the PORT
    app.run(host="0.0.0.0", port=port, debug=False)
