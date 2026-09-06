import torch
import torchvision.models as models
from PIL import Image
import numpy as np
from pathlib import Path

CLASS_NAMES = ['Cable Run', 'Center In', 'Center Out', 'Compound Flight', 
               'Downlook', 'Overall', 'Tower Flight', 'Uplook']

def load_model(model_path='flight_profile_classifier_45towers.pth'):
    device = torch.device('cpu')
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 8)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(image_path, model):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    return CLASS_NAMES[pred_idx], confidence

if __name__ == '__main__':
    model = load_model()
    profile, conf = predict_image('example.jpg', model)
    print(f"Profile: {profile}, Confidence: {conf:.2%}")
