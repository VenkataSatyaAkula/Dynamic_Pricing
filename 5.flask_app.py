from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("/home/satya6844/mysite/retail_linear_regression.pkl", "rb") as f:
    model = pickle.load(f)

# Load Scaler
with open("/home/satya6844/mysite/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/")
def home():
    return render_template("pred.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get Form Values
        product_id = float(request.form["product_id"])
        product_category_name = float(request.form["product_category_name"])
        qty = float(request.form["qty"])
        freight_price = float(request.form["freight_price"])
        product_photos_qty = float(request.form["product_photos_qty"])
        lag_price = float(request.form["lag_price"])

        # Prepare Features
        features = np.array([[
            product_id,
            product_category_name,
            qty,
            freight_price,
            product_photos_qty,
            lag_price
        ]])

        # Scale Features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        # Negative prediction avoid
        if prediction < 0:
            prediction = 0

        return render_template(
            "pred.html",
            prediction=prediction
        )

    except Exception as e:
        print(e)

        return render_template(
            "pred.html",
            prediction=None,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)