from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ============================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = Path(__file__).resolve().parents[1] / "churn_decision_tree.joblib"

try:
    MODEL = joblib.load(MODEL_PATH)
    MODEL_ERROR = None
    print("Model loaded successfully.")

except Exception as exc:
    MODEL = None
    MODEL_ERROR = str(exc)
    print("Model loading failed:", MODEL_ERROR)


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def health_check():

    if MODEL is None:
        return jsonify({
            "status": "error",
            "message": "Model failed to load",
            "error": MODEL_ERROR
        }), 500

    return jsonify({
        "status": "success",
        "message": "Telecom Churn API is running!"
    }), 200


# ============================================================
# CHURN PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if MODEL is None:
        return jsonify({
            "error": f"Model failed to load: {MODEL_ERROR}"
        }), 500

    try:

        # ----------------------------------------------------
        # GET DATA FROM FRONTEND
        # ----------------------------------------------------

        data = request.get_json() or {}

        tenure = float(data.get("tenure", 12))
        monthly_bill = float(data.get("monthlyBill", 599))
        usage_gb = float(data.get("dataUsage", 10))
        complaints = float(data.get("complaints", 0))
        contract_type = data.get("contractType", "1 Year")


        # ----------------------------------------------------
        # PREPARE FEATURES FOR THE TRAINED MODEL
        # ----------------------------------------------------

        estimated_age = min(
            70,
            max(
                18,
                18 + (tenure / 60) * 52
            )
        )

        plan_type_enc = (
            0
            if contract_type == "Month-to-Month"
            else 1
        )

        input_data = pd.DataFrame([{
            "age": estimated_age,
            "usage_gb": usage_gb,
            "complaints": complaints,
            "tenure_months": tenure,
            "plan_type_enc": plan_type_enc
        }])


        # ----------------------------------------------------
        # MACHINE LEARNING PREDICTION
        # ----------------------------------------------------

        ml_probability = MODEL.predict_proba(input_data)[0][1]

        ml_score = ml_probability * 100


        # ====================================================
        # BEHAVIORAL RISK SCORE
        #
        # This makes the system more granular because the
        # Decision Tree often returns exactly 0% or 100%.
        # ====================================================

        # Complaints contribution: maximum 30 points
        complaint_score = min(complaints / 5, 1) * 30


        # Tenure contribution: maximum 25 points
        # New customers have higher risk.
        tenure_score = max(
            0,
            min(
                1,
                (24 - tenure) / 24
            )
        ) * 25


        # Data usage contribution: maximum 15 points
        usage_score = min(
            usage_gb / 100,
            1
        ) * 15


        # Contract contribution: maximum 20 points
        contract_score = (
            20
            if contract_type == "Month-to-Month"
            else 5
        )


        # Monthly bill contribution: maximum 10 points
        bill_score = min(
            monthly_bill / 5000,
            1
        ) * 10


        # Total behavioral score
        behavioral_score = (
            complaint_score
            + tenure_score
            + usage_score
            + contract_score
            + bill_score
        )


        # ====================================================
        # FINAL HYBRID RISK SCORE
        #
        # 50% trained ML model
        # 50% behavioral risk analysis
        # ====================================================

        final_score = (
            (ml_score * 0.50)
            + (behavioral_score * 0.50)
        )


        # Keep score between 0 and 100
        final_score = max(
            0,
            min(
                100,
                final_score
            )
        )


        # ====================================================
        # RISK CLASSIFICATION
        # ====================================================

        if final_score < 15:

            label = "Low Risk"

        elif final_score < 70:

            label = "Medium Risk"

        else:

            label = "High Risk"


        # ====================================================
        # EXPLANATION / RISK FACTORS
        # ====================================================

        factors = [

            f"Complaints: {int(complaints)} calls",

            f"Tenure: {int(tenure)} months",

            f"Data Usage: {usage_gb:.1f} GB",

            f"Bill Amount: ₹{monthly_bill:.0f}",

            f"Plan Type: {contract_type}"

        ]


        # ====================================================
        # RESPONSE TO FRONTEND
        # ====================================================

        result = {

            "score": round(final_score),

            "label": label,

            "factors": factors,

            "ml_score": round(ml_score),

            "behavioral_score": round(behavioral_score)

        }


        print("Prediction:", result)


        return jsonify(result), 200


    except Exception as exc:

        print("Prediction error:", str(exc))

        return jsonify({
            "error": str(exc)
        }), 400


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )