from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def home():
    
    df = pd.read_csv('HousingData.csv')
    df=df.fillna(df.mean())
    # Load the trained model
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Serialize and save the trained model
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input values from the form
    crim = request.form["crim"]
    zn = request.form["zn"]
    indus = request.form["indus"]
    chas = request.form["chas"]
    nox = request.form["nox"]
    rm = request.form["rm"]
    age = request.form["age"]
    dis = request.form["dis"]
    rad = request.form["rad"]
    tax = request.form["tax"]
    ptratio = request.form["ptratio"]
    b = request.form["b"]
    lstat = request.form["lstat"]
    
    # Load the trained model
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Convert the input values to floats and create a numpy array
    input_values = np.array([[float(crim), float(zn), float(indus), float(chas), float(nox), float(rm), float(age), 
                              float(dis), float(rad), float(tax), float(ptratio), float(b), float(lstat)]])

    # Make a prediction using the trained model
    predicted_price = model.predict(input_values)[0]

    # Display the predicted price on the result page
    return render_template("result.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
