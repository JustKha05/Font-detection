# app.py
import numpy as np
from pathlib import Path

import streamlit as st
from PIL import Image

# ---- Keras (TF) ----
from tensorflow import keras

# ---- PyTorch ----
import torch
import torch.nn.functional as F
from torchvision import transforms

# ================== CẤU HÌNH ==================

IMG_H, IMG_W = 64, 256
CLASS_NAMES = ["display", "monospace", "san_serif", "script", "serif"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================== BUILD MODEL TORCH (CẦN SỬA CHO ĐÚNG) ==================

def build_torch_model():
    """
    TODO: SỬA LẠI HÀM NÀY ĐỂ KHỚP VỚI CODE TRAIN ViT CỦA BẠN.

    Ví dụ nếu lúc train bạn dùng timm:

        import timm
        model = timm.create_model(
            "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
            pretrained=False,
            in_chans=1,      # nếu ảnh grayscale
            num_classes=5,
        )
        return model

    Hoặc nếu bạn có class FontViT riêng:

        from vit_model import FontViT
        model = FontViT(num_classes=5)
        return model
    """
    raise NotImplementedError(
        "Hãy implement build_torch_model() giống hệt code train ViT (PyTorch) của bạn."
    )


# ================== LOAD MODEL TỔNG ==================

@st.cache_resource
def load_font_model(model_path: str):
    """
    Tự động nhận diện backend theo đuôi file:
    - .keras / .h5 -> Keras
    - .pth / .pt   -> PyTorch
    Trả về dict: {'backend': 'keras'/'torch', 'model': model}
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {p}")

    suffix = p.suffix.lower()

    # ---- Keras (.keras / .h5) ----
    if suffix in [".keras", ".h5"]:
        model = keras.models.load_model(p)
        return {"backend": "keras", "model": model}

    # ---- PyTorch (.pth / .pt) ----
    if suffix in [".pth", ".pt"]:
        model = build_torch_model()
        ckpt = torch.load(p, map_location=DEVICE)

        # Nếu bạn lưu dạng {"model_state_dict": ...}
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return {"backend": "torch", "model": model}

    # ---- Không hỗ trợ ----
    raise ValueError(
        f"Định dạng file không hỗ trợ: {suffix}. "
        "Chỉ hỗ trợ .keras, .h5, .pth, .pt."
    )


# ================== TIỀN XỬ LÝ ẢNH ==================

def preprocess_for_keras(pil_img: Image.Image):
    """Ảnh -> numpy [1, H, W, 1] cho CNN Keras."""
    img = pil_img.convert("L")
    img = img.resize((IMG_W, IMG_H))
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)   # [H, W, 1]
    arr = np.expand_dims(arr, axis=0)    # [1, H, W, 1]
    return arr, img


def preprocess_for_torch(pil_img: Image.Image):
    """Ảnh -> tensor [1, C, H, W] cho ViT / CNN PyTorch."""
    # Nếu bạn train ViT 1 kênh:
    img = pil_img.convert("L")
    img = img.resize((IMG_W, IMG_H))

    transform = transforms.Compose([
        transforms.ToTensor(),        # [C, H, W], C = 1
        # Nếu lúc train có normalize thì thêm ở đây:
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1, C, H, W]
    return tensor, img


# ================== HÀM DỰ ĐOÁN CHUNG ==================

def run_font_classifier(image_file, model_path: str):
    """
    - Đọc ảnh
    - Load model Keras hoặc Torch tùy đuôi file model_path
    - Trả về: (pred_label, probs, resized_img, backend)
    """
    pil_img = Image.open(image_file)

    model_info = load_font_model(model_path)
    backend = model_info["backend"]
    model = model_info["model"]

    if backend == "keras":
        x, resized_img = preprocess_for_keras(pil_img)
        probs = model.predict(x)[0]          # numpy [5]
    elif backend == "torch":
        x, resized_img = preprocess_for_torch(pil_img)
        with torch.no_grad():
            logits = model(x)                # [1, 5]
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    else:
        raise ValueError(f"Backend không hỗ trợ: {backend}")

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    return pred_label, probs, resized_img, backend


# ================== STREAMLIT UI ==================

def main():
    st.set_page_config(page_title="Font Family Detection", page_icon="🔤")
    st.title("🔤 Font Family Detection (5 lớp)")

    st.write(
        """
        Ứng dụng demo nhận diện **font family** từ 1 ảnh chứa chữ.
        
        Các lớp:
        - `display`
        - `monospace`
        - `san_serif`
        - `script`
        - `serif`
        """
    )

    # ----- Sidebar: cấu hình -----
    with st.sidebar:
        st.header("Cấu hình model")

        default_ckpt = "/workspaces/Font-detection/ViT/vit_font_best.pth"
        # hoặc "checkpoints/best_model.keras" tùy bạn

        model_path = st.text_input(
            "Đường dẫn checkpoint (.keras / .h5 / .pth / .pt)",
            value=default_ckpt,
        )

        st.caption(f"Thiết bị PyTorch: **{DEVICE}**")

    # ----- Upload ảnh -----
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
            with st.spinner("Đang phân loại font..."):
                pred_label, probs, resized_img, backend = run_font_classifier(
                    uploaded_img,
                    model_path=model_path,
                )
        except (FileNotFoundError, ValueError, NotImplementedError) as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Lỗi khi load model hoặc dự đoán: {e}")
            return

        st.success(f"✅ Dự đoán: **{pred_label}** (backend: {backend})")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ảnh sau khi resize:**")
            st.image(resized_img, width=256, clamp=True)

        with col2:
            st.markdown("**Xác suất từng lớp:**")
            for cls, p in zip(CLASS_NAMES, probs):
                st.write(f"- `{cls}`: {p:.4f}")

            try:
                import pandas as pd
                df = pd.DataFrame({"class": CLASS_NAMES, "prob": probs}).set_index("class")
                st.bar_chart(df)
            except ImportError:
                pass


if __name__ == "__main__":
    main()
