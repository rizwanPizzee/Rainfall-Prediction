from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

features = [
    "Location","MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine",
    "WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm",
    "WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm",
    "Pressure9am","Pressure3pm","Cloud9am","Cloud3pm",
    "Temp9am","Temp3pm","RainYesterday","Season"
]

numeric_features = {
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        data = {}
        for feat in features:
            val = request.form.get(feat)
            if val is None or val.strip() == "":
                return render_template('index.html', prediction_text=f"Please provide a value for '{feat}'")

            val = val.strip()

            if feat in numeric_features:
                try:
                    data[feat] = float(val.replace(',', ''))
                except ValueError:
                    return render_template('index.html', prediction_text=f"Invalid numeric value for '{feat}': {val}")
            else:
                data[feat] = val

        input_df = pd.DataFrame([data], columns=features)

        pred = model.predict(input_df)[0]
        pred_str = str(pred).strip().lower()

        positive = {'1', '1.0', 'yes', 'y', 'true', 'rain'}
        if pred_str in positive:
            text = " Rainfall"
        else:
            text = " No Rainfall"

        return render_template('index.html', prediction_text=text, input_data = data)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)

