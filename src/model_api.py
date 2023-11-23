import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Create an instance of FastAPI
app = FastAPI(
    title="Car Price Prediction",
    version="v1.0.0"
)

# Define a Pydantic model that represents the data structure
class CarPrice(BaseModel):
    Type: str
    Region: str
    Make: str
    Gear_Type: str
    Origin: str
    Options: str
    Year: int
    Engine_Size: float
    Mileage: int
    Price: int

# Define a Python class to create a list to reformat the data
class Item(BaseModel):
    data: List[CarPrice]

# Loading the saved model
model = pickle.load(open("F:\\Data Science Project\\Saudi Arabia Used Cars\\model\\final.sav", 'rb'))

# Create a POST endpoint to make prediction
@app.post('/prediction')
async def car_price_prediction(parameters: Item):
    # Get inputs
    req = parameters.dict()['data']

    # Convert input into Pandas DataFrame
    data = pd.DataFrame(req)

    # Make the predictions
    res = model.predict(data).tolist()

    return {"Request": req, "Response": res}

if __name__ == '__main__':
    uvicorn.run("model_api:app", host="127.0.0.1", port=8000, log_level="info", reload=True)