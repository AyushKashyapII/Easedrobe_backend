from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
import torch
import io
import uvicorn

app = FastAPI()

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

ATTRIBUTE_CATEGORIES = {
    "type": [
        "t-shirt", "shirt", "polo", "tank top", "sweater", "hoodie", "sweatshirt", "jacket",
        "coat", "blazer", "cardigan", "kurta", "sherwani", "top", "dress", "gown", "jumpsuit",
        "jeans", "trousers", "chinos", "cargo pants", "shorts", "skirt", "leggings", "joggers",
        "shrug", "overcoat", "trench coat", "parka", "puffer", "windbreaker"
    ],

    "color": [
        "black", "white", "blue", "navy", "light blue", "red", "green", "olive", "yellow",
        "orange", "pink", "purple", "grey", "brown", "beige", "cream", "maroon", "pastel",
        "teal", "mint", "lavender", "burgundy", "mustard"
    ],

    "material": [
        "cotton", "denim", "leather", "wool", "silk", "linen", "rayon", "polyester", "nylon",
        "velvet", "corduroy", "satin", "fleece", "mesh", "lace", "knit", "chiffon", "jersey", "spandex"
    ],

    "pattern": [
        "plain", "striped", "checked", "plaid", "floral", "graphic", "animal print",
        "polka dot", "abstract", "embroidered", "tie-dye", "camouflage", "color-blocked",
        "geometric", "aztec", "tribal", "ombre"
    ],

    "style": [
        "casual", "formal", "streetwear", "business", "party", "athleisure", "ethnic",
        "fusion", "boho", "vintage", "korean", "minimal", "preppy", "grunge", "punk",
        "resort", "smart casual"
    ],

    "fit": [
        "slim fit", "regular fit", "oversized", "relaxed fit", "boxy fit",
        "A-line", "bodycon", "flare", "tapered", "straight cut", "high waist", "low waist"
    ],

    "features": [
        "long sleeve", "short sleeve", "sleeveless", "half sleeve", "quarter sleeve",
        "crop", "high neck", "round neck", "v-neck", "collared", "button-up", "zip-up",
        "drawstring", "elastic waistband", "asymmetrical", "pleated", "belted"
    ],

    "target_audience": [
        "men", "women", "unisex", "boys", "girls", "kids", "teens"
    ]

}

@app.get("/")
def home():
    return {"message": "Fashion AI"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

        attributes = {}
        for category, labels in ATTRIBUTE_CATEGORIES.items():
            result = classifier(caption, candidate_labels=labels, multi_label=True)

            if category == "features":
                selected = [label for i, label in enumerate(result["labels"][:3]) if result["scores"][i] > 0.4]
                attributes[category] = selected if selected else ["unknown"]

            elif category in ["color", "material", "pattern", "style"]:
                selected = [label for i, label in enumerate(result["labels"][:2]) if result["scores"][i] > 0.4]
                attributes[category] = selected if selected else ["unknown"]

            else:
                top_label = result["labels"][0]
                top_score = result["scores"][0]
                attributes[category] = top_label if top_score > 0.4 else "unknown"

        return JSONResponse(content={
            "caption": caption,
            "attributes": attributes
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
