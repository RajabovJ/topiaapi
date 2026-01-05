from fastapi import FastAPI, File, UploadFile
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import io
import torch.nn.functional as F

app = FastAPI(
    title="Tomato Disease Classification API",
    description="Pomidor kasalliklarini tasniflovchi FastAPI backend",
    version="1.0.0",
)

model_path = "rajabovv/tomato-disease-classifier/tomatoDiseaseClassifier"
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)  # ehtimollarni hisoblash
            predicted_idx = probs.argmax(-1).item()
            confidence = probs[0, predicted_idx].item()  # aniqlik darajasi

        labels = model.config.id2label
        predicted_label = labels[predicted_idx]

        return {
            "success": True,
            "predicted_id": predicted_idx,
            "predicted_label": predicted_label,
            "confidence": confidence
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
