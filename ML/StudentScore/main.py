from fastapi import FastAPI
import joblib
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
