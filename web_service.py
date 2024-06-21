
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# Define the input data structure using Pydantic
class InputData(BaseModel):
    input: List[int]  # A list of integers

app = FastAPI()

# Load mlp model
model = joblib.load('model_mlp.pkl')

# Function to convert integers to binary representation
def integer_to_binary(input_list):
    return [[int(bit) for bit in format(i, '012b')] for i in input_list]

# Mapping from numeric labels to string labels
label_map = {0: 'None', 1: 'Fizz', 2: 'Buzz', 3: 'FizzBuzz'}

@app.post('/predict')
async def predict(data: InputData):
    # Convert the input integers to binary representation
    binary_input = integer_to_binary(data.input)

    # Perform the prediction with the model
    prediction = model.predict(binary_input)

    # Decode the numeric predictions to string labels
    decoded_prediction = [label_map[p] for p in prediction]

    # Format the response as JSON
    response = {
        'prediction': decoded_prediction
    }

    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host= 'localhost', port=8004)