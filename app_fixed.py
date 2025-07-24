
import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import os
import tempfile
import numpy as np
from torchvision import models
from PIL import Image
from collections import defaultdict

# ========== Konfigurasi model ==========
RESNEXT_MODEL_PATH = "hasil_model_resnext_fc.pt"
LSTM_MODEL_PATH = "hasil_train_model.pt"
SEQUENCE_LENGTH = 20

# ========== Load model ResNeXt feature extractor ==========
@st.cache_resource
def load_feature_extractor():
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
    return torch.nn.Sequential(*list(model.children())[:-1]).eval()

# ========== Load model FC (ResNeXt only) ==========
class SimpleFC(torch.nn.Module):
    def __init__(self, input_size=2048, hidden_size=512):
        super(SimpleFC, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.classifier(x).squeeze()

def load_resnext_fc_model():
    model = SimpleFC()
    model.load_state_dict(torch.load(RESNEXT_MODEL_PATH))
    model.eval()
    return model

# ========== Load model LSTM ==========
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, num_layers=2, num_classes=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out.squeeze()

def load_lstm_model():
    model = LSTMClassifier()
    model.load_state_dict(torch.load(LSTM_MODEL_PATH))
    model.eval()
    return model

# ========== Ekstrak frame dari video ==========
def extract_frames(video_path, frame_skip=5, max_frames=500):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1
    cap.release()
    return frames

# ========== Transformasi gambar ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== Prediksi dengan ResNeXt Only ==========
def predict_resnext_only(frames, feature_extractor, model_fc):
    features = []
    for img in frames:
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = feature_extractor(tensor).view(1, -1)
            features.append(feat)
    features_tensor = torch.cat(features)
    with torch.no_grad():
        outputs = model_fc(features_tensor)
        preds = torch.sigmoid(outputs) > 0.5
    majority = torch.mode(preds.int()).values.item()
    return majority, len(frames), None  # label, total frame, None for sequence

# ========== Prediksi dengan ResNeXt + LSTM ==========
def predict_resnext_lstm(frames, feature_extractor, lstm_model):
    features = []
    for img in frames:
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = feature_extractor(tensor).view(1, -1)
            features.append(feat)
    features_tensor = torch.cat(features)

    sequences = []
    for i in range(0, len(features_tensor) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
        chunk = features_tensor[i:i + SEQUENCE_LENGTH]
        sequences.append(chunk.unsqueeze(0))  # [1, 20, 2048]

    if not sequences:
        return -1, len(frames), 0  # Tidak cukup frame

    input_seq = torch.cat(sequences)  # [N, 20, 2048]
    with torch.no_grad():
        outputs = lstm_model(input_seq)
        preds = torch.sigmoid(outputs) > 0.5
    majority = torch.mode(preds.int()).values.item()
    return majority, len(frames), len(sequences)

# ========== Tampilan Streamlit ==========
st.title("🔍 Deteksi Video Deepfake")
st.write("Unggah video MP4 untuk mendeteksi apakah itu **Real** atau **Fake**.")

uploaded_file = st.file_uploader("📤 Upload Video (.mp4)", type=["mp4"])
model_choice = st.selectbox("Pilih Model Deteksi", ["ResNeXt Only", "ResNeXt + LSTM"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.video(tfile.name)

    st.info("📥 Video diterima. Mengekstrak frame...")
    frames = extract_frames(tfile.name)
    st.success(f"✅ Berhasil ambil {len(frames)} frame.")

    feature_extractor = load_feature_extractor()

    if model_choice == "ResNeXt Only":
        model = load_resnext_fc_model()
        label, total_frame, total_seq = predict_resnext_only(frames, feature_extractor, model)
    else:
        model = load_lstm_model()
        label, total_frame, total_seq = predict_resnext_lstm(frames, feature_extractor, model)

    if label == -1:
        st.warning("⚠️ Tidak cukup frame untuk membentuk sequence.")
    else:
        result = "FAKE" if label == 1 else "REAL"
        st.markdown(f"### 🎯 Hasil Prediksi: `{result}`")
        st.write(f"Total Frame: {total_frame}")
        if model_choice == "ResNeXt + LSTM" and total_seq is not None:
            st.write(f"Total Sequence: {total_seq}")
