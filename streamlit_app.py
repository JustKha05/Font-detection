# app.py
import numpy as np
from pathlib import Path

import streamlit as st
from tensorflow import keras
from PIL import Image

# ================== CẤU HÌNH ==================

IMG_H, IMG_W = 64, 256
CLASS_NAMES = ["display", "monospace", "san_serif", "script", "serif"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "CNN" / "best_model.keras"


@st.cache_resource
def load_font_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_img: Image.Image):
    img = pil_img.convert("L")
    img = img.resize((IMG_W, IMG_H))

    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)   # [H, W, 1]
    arr = np.expand_dims(arr, axis=0)    # [1, H, W, 1]
    return arr


def run_font_classifier(image_file):
    """
    Nhận file ảnh upload, trả về (pred_label, probs)
    """
    pil_img = Image.open(image_file)
    x = preprocess_image(pil_img)

    model = load_font_model()

    probs = model.predict(x)[0]          # shape (5,)
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    return pred_label, probs


# ================== STREAMLIT UI ==================

def main():
    st.set_page_config(page_title="Font Family Detection", page_icon="🔤")

    hide_sidebar_style = """
        <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)

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
            with st.spinner("Đang phân loại font..."):
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
