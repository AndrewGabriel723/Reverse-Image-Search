from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from io import BytesIO
app = FastAPI()

# Load the class mappings
with open('./model/class_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

# Load the model and set it to evaluation mode
resnet_model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
resnet_model.eval()

# Define the transformation for incoming images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the API endpoint for prediction
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            out = resnet_model(image)
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        top_5_predictions = [(mappings[idx.item()], percentage[idx].item()) for idx in indices[0][:5]]
        max_prob_class = top_5_predictions[0][0]
        return JSONResponse(content={"Identified Image": max_prob_class}, status_code=200)
    except:
        return JSONResponse(content="Invalid image format or error in prediction.", status_code=400)
