# Copyright 2025 Alireza Aghamohammadi

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conint

from src.config_reader import read_config

config_path = "config/config.yaml"
web_config = read_config(config_path)["web"]

model_dir = web_config["model_output_dir"]
model_name = web_config["model_name"]


model = joblib.load(Path(model_dir) / model_name)

app = FastAPI()


class DemandRequest(BaseModel):
    hour_of_day: conint(ge=0, le=23)
    day: conint(ge=0)
    row: conint(ge=0, le=7)
    col: conint(ge=0, le=7)


class DemandResponse(BaseModel):
    demand: int


@app.post("/predict", response_model=DemandResponse)
def predict_demand(request: DemandRequest):
    features = pd.DataFrame(
        [
            {
                "hour_of_day": request.hour_of_day,
                "day": request.day,
                "row": request.row,
                "col": request.col,
            }
        ]
    )

    prediction = model.predict(features)[0]
    return {"demand": round(prediction)}


if __name__ == "__main__":
    uvicorn.run(
        "web.application:app",
        host=os.environ.get("WEB_HOST", web_config["host"]),
        port=int(os.environ.get("WEB_PORT", web_config["port"])),
        reload=True,
    )
