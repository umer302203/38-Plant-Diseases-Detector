import gradio as gr
import numpy as np
import json
import requests
import io
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

IMG_SIZE = (224, 224)
NUM_CLASSES = 38

base_model = MobileNetV2(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.load_weights('plant_disease_mobilenet.keras')
print("✅ Weights loaded successfully.")

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

display_names = {
    "Apple___Apple_scab": "Apple - Apple Scab",
    "Apple___Black_rot": "Apple - Black Rot",
    "Apple___Cedar_apple_rust": "Apple - Cedar Apple Rust",
    "Apple___healthy": "Apple - Healthy",
    "Blueberry___healthy": "Blueberry - Healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry - Powdery Mildew",
    "Cherry_(including_sour)___healthy": "Cherry - Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn - Cercospora Leaf Spot / Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Corn - Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn - Northern Leaf Blight",
    "Corn_(maize)___healthy": "Corn - Healthy",
    "Grape___Black_rot": "Grape - Black Rot",
    "Grape___Esca_(Black_Measles)": "Grape - Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape___healthy": "Grape - Healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange - Huanglongbing (Citrus Greening)",
    "Peach___Bacterial_spot": "Peach - Bacterial Spot",
    "Peach___healthy": "Peach - Healthy",
    "Pepper,_bell___Bacterial_spot": "Bell Pepper - Bacterial Spot",
    "Pepper,_bell___healthy": "Bell Pepper - Healthy",
    "Potato___Early_blight": "Potato - Early Blight",
    "Potato___Late_blight": "Potato - Late Blight",
    "Potato___healthy": "Potato - Healthy",
    "Raspberry___healthy": "Raspberry - Healthy",
    "Soybean___healthy": "Soybean - Healthy",
    "Squash___Powdery_mildew": "Squash - Powdery Mildew",
    "Strawberry___Leaf_scorch": "Strawberry - Leaf Scorch",
    "Strawberry___healthy": "Strawberry - Healthy",
    "Tomato___Bacterial_spot": "Tomato - Bacterial Spot",
    "Tomato___Early_blight": "Tomato - Early Blight",
    "Tomato___Late_blight": "Tomato - Late Blight",
    "Tomato___Leaf_Mold": "Tomato - Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato - Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato___Target_Spot": "Tomato - Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato - Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato - Mosaic Virus",
    "Tomato___healthy": "Tomato - Healthy"
}

supported_crops = [
    "🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn", "🍇 Grape",
    "🍊 Orange", "🍑 Peach", "🫑 Bell Pepper", "🥔 Potato", "🍓 Raspberry",
    "🫘 Soybean", "🎃 Squash", "🍓 Strawberry", "🍅 Tomato"
]
crops_text = "\n".join(supported_crops)

disease_links = []
for name in sorted(display_names.values()):
    query = name.replace(" ", "+")
    url = f"https://www.google.com/search?tbm=isch&q={query}"
    disease_links.append(f"<a href='{url}' target='_blank'>{name}</a>")
disease_links_html = " | ".join(disease_links)

API_URL = Insert your here
BLIP_TOKEN = Insert your here
HEADERS = Insert your here

def predict_disease(image):
    img = image.resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr, verbose=0)[0]
    idx = str(np.argmax(preds))
    original_name = class_names[idx]
    disease = display_names.get(original_name, original_name)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    response = requests.post(API_URL, headers=HEADERS, data=buffered.getvalue())
    if response.status_code == 200:
        caption = response.json()[0]['generated_text']
    else:
        caption = "Caption unavailable"

    return disease, caption

with gr.Blocks(title="Plant Disease Doctor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌿 AI Plant Disease Doctor (MobileNetV2 + BLIP)")
    gr.Markdown("**Please upload a leaf image of one of the crops listed below.** The AI will identify the disease and describe the visual symptoms.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Leaf Image")
            btn = gr.Button("Diagnose", variant="primary")
        with gr.Column(scale=1):
            disease_output = gr.Textbox(label="Detected Disease", interactive=False)
            caption_output = gr.Textbox(label="AI Visual Description", interactive=False)

    gr.Markdown("### 📋 Supported Crops (only these plants)")
    gr.Markdown(crops_text)

    gr.Markdown("### 🔍 Search Sample Images for Each Disease")
    gr.HTML(disease_links_html)

    btn.click(fn=predict_disease, inputs=image_input, outputs=[disease_output, caption_output])

demo.launch()
