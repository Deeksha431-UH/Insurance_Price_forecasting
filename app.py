from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open("./gradient_boosting_regressor_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Load dataset for dropdowns
data = pd.read_csv('./clean_data.csv')

@app.route('/')
def index():
    sex = sorted(data['sex'].unique())
    smoker = sorted(data['smoker'].unique())
    region = sorted(data['region'].unique())
    return render_template('index.html', sex=sex, smoker=smoker, region=region)

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    # Model prediction
    prediction = model.predict(pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))

    predicted_amount = prediction[0]  # raw predicted amount

    # Total insurance is always fixed at 1,00,000
    total_insurance = 100000

    # Premium calculation (5% of predicted amount, capped at total insurance)
    capped_amount = min(predicted_amount, total_insurance)
    premium_rate = 0.05
    premium = capped_amount * premium_rate

    # Return all three values separated by "|"
    return f"{predicted_amount}|{total_insurance}|{premium}"

if __name__ == "__main__":
    app.run(debug=True)
