from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


# ==================================================
# Test Root Endpoint
# ==================================================

def test_home():

    response = client.get("/")

    assert response.status_code == 200
    assert "E-Commerce Purchase Predictor" in response.text


# ==================================================
# Test Health Endpoint
# ==================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ==================================================
# Test Model Information
# ==================================================

def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "XGBoost"
    assert data["threshold"] == 0.35
    assert data["roc_auc"] > 0.9
    assert data["f1"] > 0.6


# ==================================================
# Test Prediction Endpoint
# ==================================================

def test_prediction():

    payload = {

        "Administrative": 3,
        "Administrative_Duration": 100,

        "Informational": 2,
        "Informational_Duration": 50,

        "ProductRelated": 20,
        "ProductRelated_Duration": 1000,

        "BounceRates": 0.02,
        "ExitRates": 0.05,

        "PageValues": 10,
        "SpecialDay": 0,

        "Month": "Nov",

        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,

        "VisitorType": "Returning_Visitor",

        "Weekend": False
    }


    response = client.post(
        "/predict",
        json=payload
    )


    assert response.status_code == 200


    data = response.json()


    assert "purchase_probability" in data
    assert "threshold" in data
    assert "will_purchase" in data


    assert 0 <= data["purchase_probability"] <= 1

    assert data["threshold"] == 0.35

    assert isinstance(
        data["will_purchase"],
        bool
    )


# ==================================================
# Test Invalid Request
# ==================================================

def test_invalid_prediction():

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422