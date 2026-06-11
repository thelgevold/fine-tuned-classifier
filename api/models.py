from typing import Literal

from pydantic import BaseModel, Field


class CategorizeQuestionRequest(BaseModel):
    model_name: str = Field(..., min_length=1, description="Ollama model name to query")
    question: str = Field(..., min_length=1, description="Homeowner question to categorize")
    label_mode: Literal["code", "category"] = Field(
        default="code",
        description="Whether the target model was trained to emit opaque codes or full category names",
    )
    num_predict: int = Field(
        default=4,
        ge=1,
        description="Maximum number of tokens to generate for the category response",
    )
    think: bool = Field(
        default=False,
        description="Whether to allow the model to use thinking mode during generation",
    )


class CategorizeQuestionResponse(BaseModel):
    model_name: str
    question: str
    code: str
    category: str


class AvailableModel(BaseModel):
    name: str
    size: int | None = None
    modified_at: str | None = None


class AvailableModelsResponse(BaseModel):
    models: list[AvailableModel]
