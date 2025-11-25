from typing import List
from pydantic import BaseModel, Field


class Request(BaseModel):
    model_key: str = Field(..., description="The key of the model.")
    user_id: str = Field(..., description="The user id of the model.")
    seeds: List[List[int]] = Field(..., description="The instance seeds for the inference.")