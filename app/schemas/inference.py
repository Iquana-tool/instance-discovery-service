from typing import Any
from pydantic import BaseModel, Field


class Request(BaseModel):
    model_key: str = Field(..., description="The key of the model.")
    user_id: str = Field(..., description="The user id of the model.")
    seeds: list[list[list[bool]]] = Field(...,
                                                description="Seeds is a list of binary masks.")