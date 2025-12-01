import numpy as np
from pathlib import Path

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

IMG_H, IMG_W = 64, 256
CLASS_NAMES = ["display", "monospace", "san_serif", "script", "serif"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "CNN" / "best_custom_cnn_weights.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flattened_features_size = 128 * 8 * 32
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_features_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@st.cache_resource
def load_font_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")
    model = CustomCNN(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


test_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def preprocess_image(pil_img: Image.Image):
    tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)
    return tensor


def run_font_classifier(image_file):
    pil_img = Image.open(image_file).convert("RGB")
    x = preprocess_image(pil_img)
    model = load_font_model()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    return pred_label, probs


def main():
    st.set_page_config(
        page_title="Font Family Detection – CNN",
        page_icon="🔤",
        layout="centered",
    )

    custom_css = """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        .main-block {
            max-width: 850px;
            margin: 0 auto;
            padding: 1.5rem 1.5rem 2rem 1.5rem;
            border-radius: 18px;
            background: rgba(17, 24, 39, 0.85);
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
            border: 1px solid rgba(75, 85, 99, 0.6);
        }

        .app-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }

        .app-subtitle {
            font-size: 0.95rem;
            color: #9CA3AF;
            margin-bottom: 1.2rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
            background: rgba(59, 130, 246, 0.15);
            color: #BFDBFE;
            border: 1px solid rgba(59, 130, 246, 0.4);
            margin-bottom: 0.8rem;
        }

        .upload-label {
            font-weight: 600;
            margin-bottom: 0.3rem;
        }

        .result-card {
            margin-top: 1.2rem;
            padding: 1rem 1.1rem;
            border-radius: 14px;
            background: rgba(15, 118, 110, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.35);
        }

        .result-title {
            font-size: 0.9rem;
            color: #6EE7B7;
            margin-bottom: 0.3rem;
        }

        .result-label {
            font-size: 1.4rem;
            font-weight: 700;
        }

        .probs-title {
            font-size: 0.9rem;
            margin-top: 1.2rem;
            margin-bottom: 0.4rem;
            font-weight: 600;
        }

        .prob-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.4rem;
            gap: 0.4rem;
        }

        .prob-label {
            flex: 0 0 90px;
            font-size: 0.85rem;
            text-transform: lowercase;
            color: #E5E7EB;
        }

        .prob-bar {
            flex: 1;
            height: 0.45rem;
            border-radius: 999px;
            background: rgba(31, 41, 55, 0.9);
            overflow: hidden;
        }

        .prob-bar-inner {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #3B82F6, #22C55E);
        }

        .prob-value {
            flex: 0 0 52px;
            text-align: right;
            font-size: 0.8rem;
            color: #D1D5DB;
        }

        .device-badge {
            font-size: 0.75rem;
            color: #9CA3AF;
            margin-top: 0.4rem;
        }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown("<div class='main-block'>", unsafe_allow_html=True)

    st.markdown(
        "<div class='app-title'>🔤 Font Family Detection</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='app-subtitle'>Nhận diện 5 loại font: display · monospace · "
        "san_serif · script · serif</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr />", unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Ảnh font cần phân loại",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_img is not None:
        st.image(uploaded_img, use_container_width=True)

    col_btn, _ = st.columns([1, 1])
    with col_btn:
        predict_clicked = st.button("🚀 Dự đoán font family", use_container_width=True)

    if predict_clicked:
        if uploaded_img is None:
            st.error("Bạn cần tải lên một ảnh trước.")
        else:
            try:
                with st.spinner("Đang phân loại font..."):
                    pred_label, probs = run_font_classifier(uploaded_img)
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Lỗi khi load model hoặc dự đoán: {e}")
            else:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='result-title'>Kết quả dự đoán</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='result-label'>{pred_label}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='probs-title'>Xác suất từng lớp</div>", unsafe_allow_html=True)
                for cls, p in zip(CLASS_NAMES, probs):
                    width = int(p * 100)
                    st.markdown(
                        f"""
                        <div class="prob-row">
                            <div class="prob-label">{cls}</div>
                            <div class="prob-bar">
                                <div class="prob-bar-inner" style="width: {width}%;"></div>
                            </div>
                            <div class="prob-value">{p*100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
