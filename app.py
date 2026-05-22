from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form["email_text"]

    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]

    if prediction == 1:
        result = "Spam / Phishing Email"
    else:
        result = "Not Spam"

    return render_template(
        "result.html",
        email=email_text,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)

