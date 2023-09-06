from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('your_model_file.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = float(request.form['age'])
        gender = request.form['gender']
        subscription_length = float(request.form['subscription_length'])
        monthly_bill = float(request.form['monthly_bill'])
        total_usage_gb = float(request.form['total_usage_gb'])

        # Perform any necessary preprocessing on the input data
        # For example, convert gender to numerical values

        # Make predictions using your trained model
        input_data = [[age, gender, subscription_length, monthly_bill, total_usage_gb]]
        prediction = model.predict(input_data)[0]

        # Return the prediction to the HTML template
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
