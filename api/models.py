from pydantic import BaseModel, Field


class CategorizeQuestionRequest(BaseModel):
    model_name: str = Field(..., min_length=1, description="Ollama model name to query")
    question: str = Field(..., min_length=1, description="Homeowner question to categorize")


class CategorizeQuestionResponse(BaseModel):
    model_name: str
    question: str
    code: str
    category: str
