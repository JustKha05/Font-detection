# app.py
import numpy as np
from pathlib import Path

import streamlit as st
from PIL import Image

# ---- PyTorch ----
import torch
import torch.nn.functional as F
from torchvision import transforms

# ================== CẤU HÌNH ==================

# PHẢI khớp với IMAGE_SIZE lúc train ViT
IMAGE_SIZE = 224

CLASS_NAMES = ["display", "monospace", "san_serif", "script", "serif"]

BASE_DIR = Path(__file__).resolve().parent

# Đường dẫn tới file TorchScript bạn đã save:
# scripted.save("vit_font_best_scripted.pt")
MODEL_PATH = BASE_DIR / "ViT" / "vit_font_best_scripted.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================== LOAD MODEL (TorchScript) ==================

@st.cache_resource
def load_font_model():
    """
    Load TorchScript model (.pt) để infer.
    Không cần build lại kiến trúc VitFeatureClassifier.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

    model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
    model.eval()
    return model


# ================== TIỀN XỬ LÝ ẢNH ==================

# PHẢI giống hệt transform dùng cho test_loader lúc train:
# Resize -> Grayscale(num_output_channels=3) -> ToTensor -> Normalize(...)
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image(pil_img: Image.Image):
    """
    PIL.Image -> tensor [1, 3, IMAGE_SIZE, IMAGE_SIZE] trên DEVICE
    """
    tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)
    return tensor


# ================== HÀM DỰ ĐOÁN ==================

def run_font_classifier(image_file):
    """
    Nhận file ảnh upload, trả về (pred_label, probs)
    """
    pil_img = Image.open(image_file).convert("RGB")  # để chắc chắn đọc được mọi định dạng
    x = preprocess_image(pil_img)

    model = load_font_model()

    with torch.no_grad():
        out = model(x)
        # Nếu model.forward trả về (logits, feats) như VitFeatureClassifier
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # shape (5,)

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    return pred_label, probs


# ================== STREAMLIT UI ==================

def main():
    st.set_page_config(page_title="Font Family Detection (ViT)", page_icon="🔤")

    # Ẩn sidebar
    hide_sidebar_style = """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)

    st.title("🔤 Font Family Detection (5 lớp) – ViT")

    st.write(
        """
        Ứng dụng demo nhận diện **font family** từ 1 ảnh chứa chữ (ViT, TorchScript).
        
        Các lớp:
        - `display`
        - `monospace`
        - `san_serif`
        - `script`
        - `serif`
        """
    )

    # ----- Upload ảnh gốc -----
    uploaded_img = st.file_uploader(
        "Tải lên ảnh font (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_img is not None:
        st.subheader("Ảnh gốc")
        st.image(uploaded_img, use_container_width=True)

    # ----- Nút dự đoán -----
    if st.button("🚀 Dự đoán font family"):
        if uploaded_img is None:
            st.error("Bạn cần tải lên một ảnh trước.")
            return

        try:
            with st.spinner("Đang phân loại font (ViT)..."):
                pred_label, probs = run_font_classifier(uploaded_img)
        except FileNotFoundError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Lỗi khi load model hoặc dự đoán: {e}")
            return

        st.success(f"✅ Dự đoán: **{pred_label}**")

        st.markdown("**Xác suất từng lớp:**")
        for cls, p in zip(CLASS_NAMES, probs):
            st.write(f"- `{cls}`: {p:.4f}")


if __name__ == "__main__":
    main()
