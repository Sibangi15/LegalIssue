from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()


classifier = pipeline(
    "text-classification",
    model="Sibangi01/legal_issue",
    tokenizer="Sibangi01/legal_issue",
    device=-1,
    torch_dtype=torch.float32
)


CATEGORY_TO_DEPARTMENT = {

    "Banking": "Finance",

    "Credit Reporting": "Credit Bureau",

    "Debt Collection": "Legal",

    "Loans": "Finance",

    "Fraud": "Cyber Crime",

    "Credit Card": "Finance",

    "Consumer Issues": "Consumer Affairs"
}


HIGH_URGENCY_WORDS = [
    "fraud",
    "scam",
    "harassment",
    "threat",
    "illegal",
    "stolen",
    "hack"
]


class ComplaintRequest(BaseModel):
    complaintText: str


def detect_urgency(text):

    text = text.lower()

    for word in HIGH_URGENCY_WORDS:

        if word in text:
            return "High"

    if len(text) > 250:
        return "Medium"

    return "Low"


@app.get("/")
def home():

    return {
        "message": "AI Legal Assistant API Running"
    }


@app.post("/predict")
def predict(request: ComplaintRequest):

    text = request.complaintText

    result = classifier(text)[0]

    category = result['label']

    department = CATEGORY_TO_DEPARTMENT.get(
        category,
        "General"
    )

    urgency = detect_urgency(text)

    return {

        "category": category,

        "department": department,

        "urgency": urgency,

        "score": round(float(result['score']), 2)
    }