from fastapi import APIRouter, Depends,   Body
from controllers.auth import authenticate_user, create_access_token
from config import settings
from pydantic import BaseModel
from middlewares.auth_middleware import get_current_user
from controllers.prediction import fetch_crypto_data ,add_technical_indicators
import joblib

from pydantic import BaseModel
# Load the trained model
# model = joblib.load("BTC-USD_xgboost_model.pkl")


# Define request model
class PredictionRequest(BaseModel):
    symbol: str = 'BTC-USD'  # Default to 'BTC-USD' if no symbol is provided
router = APIRouter()
@router.post("/predict")
async def predict(request: PredictionRequest , current_user: dict = Depends(get_current_user) ):
    try:
        username = current_user["username"]
        symbol = request.symbol
        print(symbol)
        model = joblib.load(f"{symbol}_xgboost_model.pkl")
        df = fetch_crypto_data(symbol, "1h", 100)
        print("1",df)
        df = add_technical_indicators(df)
        print("2",df)
        # X = df[['SMA']].values
        X = df.iloc[-1:].drop(columns=['close'], errors='ignore')
        print("3",X)
        # X = df.iloc[-1:].drop(columns=['close'], errors='ignore') 
        print("4",X)# Ensure 'close' exists before dropping
        prob = model.predict_proba(X)[0][1] * 100
        print(f"Probability of Increase for {symbol}: {prob:.2f}%")
        return {"predicted_probability": f"{prob:.2f}"}
    except Exception as e:
        print(str(e))
        return {"error": str(e)}