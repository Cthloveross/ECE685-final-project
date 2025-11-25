from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CONFIG


@dataclass(slots=True)
class ToxicityScore:
    probability: float
    label: int


class ToxicityWrapper:
    def __init__(self, model_name: str | None = None):
        model_id = model_name or CONFIG.model.toxicity_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()

    @torch.inference_mode()
    def score(self, text: str, threshold: float = 0.5) -> ToxicityScore:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        prob = probs[0, 1].item()
        return ToxicityScore(probability=prob, label=int(prob >= threshold))
