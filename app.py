import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
import numpy as np
import zipfile
import io
from pathlib import Path

CLASS_NAMES = ['Cable Run', 'Center In', 'Center Out', 'Compound Flight', 'Downlook', 'Overall', 'Tower Flight', 'Uplook']

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    if st.session_state.password_correct:
        return True
    
    st.title("Flight Profile Classifier - Login")
    st.write("Enter password to access the photo sorting tool")
    
    password = st.text_input("Password:", type="password")
    
    if password == "drone123":
        st.session_state.password_correct = True
        st.rerun()
    elif password:
        st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 8)
    model.load_state_dict(torch.load('flight_profile_classifier_45towers.pth', map_location=device))
    model.eval()
    return model

def predict_image(image_file, model):
    img = Image.open(image_file).convert('RGB').resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    
    return CLASS_NAMES[pred_idx], confidence

st.set_page_config(page_title="Flight Profile Classifier", layout="wide")
st.title("Flight Profile Classifier")
st.write("Automatically sort drone photos into 8 flight profiles")

model = load_model()

uploaded_files = st.file_uploader("Choose drone photos", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    st.write(f"📸 {len(uploaded_files)} photos uploaded")
    
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        profile, confidence = predict_image(file, model)
        
        if profile not in results:
            results[profile] = []
        results[profile].append((file.name, file.getvalue(), confidence))
        
        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}...")
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.empty()
    
    st.subheader("Results")
    for profile in CLASS_NAMES:
        if profile in results:
            st.write(f"**{profile}** - {len(results[profile])} photos")
    
    st.divider()
    st.subheader("Save Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download as ZIP"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                for profile in CLASS_NAMES:
                    if profile in results:
                        for filename, file_data, _ in results[profile]:
                            zf.writestr(f"{profile}/{filename}", file_data)
            zip_buffer.seek(0)
            st.download_button(
                label="Click here to download ZIP",
                data=zip_buffer.getvalue(),
                file_name="sorted_drone_photos.zip",
                mime="application/zip"
            )
    
    with col2:
        if st.button("Save to Computer"):
            output_dir = Path("C:/Users/RickHenriquez/sorted_photos")
            output_dir.mkdir(exist_ok=True, parents=True)
            
            saved_count = 0
            for profile in CLASS_NAMES:
                if profile in results:
                    profile_dir = output_dir / profile
                    profile_dir.mkdir(exist_ok=True)
                    
                    for filename, file_data, _ in results[profile]:
                        filepath = profile_dir / filename
                        with open(filepath, 'wb') as f:
                            f.write(file_data)
                        saved_count += 1
            
            st.success(f"Saved {saved_count} photos to sorted_photos folder")