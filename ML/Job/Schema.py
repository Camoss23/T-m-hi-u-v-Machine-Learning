from pydantic import BaseModel, Field, field_validator, ValidationError
import re

class JobInput(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    industry: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=100)
    function: str = Field(min_length=1, max_length=100)

    @field_validator("title", "industry", "location", "function")
    @classmethod
    def only_letters(cls, v):
        if not re.match(r"^[a-zA-Z\s]+$", v):
            raise ValueError("Chỉ được nhập chữ, không nhập số hoặc ký tự đặc biệt")
        return v.strip()
class PredictionOutput(BaseModel):
    prediction: str