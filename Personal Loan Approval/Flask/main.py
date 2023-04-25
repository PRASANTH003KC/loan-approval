from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas


# Create a Flask web application instance
app = Flask(__name__, template_folder='templates')

# Load the trained model from a saved pickle file
model = pickle.load(open(r'sd.pkl', 'rb'))
data=pandas.read_csv(r'train.csv')
# Define a route to render the home page HTML template
@app.route('/')
def home():
    return render_template('home.html')

# Define a route to render the input HTML form
@app.route('/index')
def predict():
    return render_template("predict.html")

# Define a route to handle form submission and display the prediction result
@app.route('/submit', methods=["POST","GET"])
def submit():
    # Read the input values submitted by the user
    input_feature = [int(x) for x in request.form.values()]
    # Convert the input values to a NumPy array
    input_feature = [np.array(input_feature)]
    # Define the column names for the input data frame
    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
       'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
    # Create a Pandas data frame from the input values with the column names
    data = pandas.DataFrame(input_feature, columns=names)

    # Use the loaded model to make a prediction
    prediction = model.predict(data)
    # Convert the prediction from a NumPy array to an integer
    prediction = int(prediction)

    # Render the output HTML template with the prediction result
    if prediction == 0:
       return render_template("result.html",result="Loan will not be approved")
    else:
       return render_template("result.html",result="Loan will be approved")

# Start the Flask web application on port 5000
if __name__ == "__main__":
    app.run(port=5000, debug=True)
