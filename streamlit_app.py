# app.py
import numpy as np
from pathlib import Path

import streamlit as st
from tensorflow import keras
from PIL import Image

# ================== CẤU HÌNH ==================

# Kích thước ảnh đúng với lúc train: (64, 256, 1)
IMG_H, IMG_W = 64, 256

# Thứ tự lớp phải KHỚP với class_indices lúc train:
# {'display': 0, 'monospace': 1, 'san_serif': 2, 'script': 3, 'serif': 4}
CLASS_NAMES = ["display", "monospace", "san_serif", "script", "serif"]


@st.cache_resource
def load_font_model(model_path: str):
    """
    Load checkpoint Keras (full model .keras/.h5).
    Dùng cache_resource để chỉ load 1 lần.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(pil_img: Image.Image):
    """
    Tiền xử lý ảnh:
    - Convert sang grayscale (1 kênh)
    - Resize về (64, 256)
    - Scale về [0,1]
    - Thêm batch dimension -> (1, 64, 256, 1)
    """
    # Grayscale 1 kênh
    img = pil_img.convert("L")  # "L" = 8-bit pixels, black and white
    # Resize: PIL resize nhận (width, height)
    img = img.resize((IMG_W, IMG_H))

    arr = np.array(img, dtype="float32") / 255.0  # [H, W]
    arr = np.expand_dims(arr, axis=-1)            # [H, W, 1]
    arr = np.expand_dims(arr, axis=0)             # [1, H, W, 1]

    return arr, img


def run_font_classifier(image_file, model_path: str):
    """
    Hàm "giống run_zipvoice" nhưng cho font classifier:
    - Nhận file ảnh upload
    - Load model từ checkpoint
    - Trả về: (pred_label, probs, ảnh đã resize)
    """
    # Đọc ảnh từ file uploader (BytesIO)
    pil_img = Image.open(image_file)

    # Tiền xử lý
    x, resized_img = preprocess_image(pil_img)

    # Load model
    model = load_font_model(model_path)

    # Dự đoán
    probs = model.predict(x)[0]         # shape (5,)
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    return pred_label, probs, resized_img


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
        # Đường dẫn checkpoint (mặc định cùng thư mục)
        default_ckpt = "best_model.keras"  # sửa lại nếu bạn để nơi khác
        model_path = st.text_input(
            "Đường dẫn checkpoint (.keras / .h5)",
            value=default_ckpt,
            help="Ví dụ: best_model.keras hoặc checkpoints/best_model.keras",
        )

    # ----- Upload ảnh -----
    uploaded_img = st.file_uploader(
        "Tải lên ảnh font (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_img is not None:
        # Hiển thị ảnh gốc
        st.subheader("Ảnh gốc")
        st.image(uploaded_img, use_column_width=True)

    # ----- Nút dự đoán -----
    if st.button("🚀 Dự đoán font family"):
        if uploaded_img is None:
            st.error("Bạn cần tải lên một ảnh trước.")
            return

        # Chạy inference
        try:
            with st.spinner("Đang phân loại font..."):
                pred_label, probs, resized_img = run_font_classifier(
                    uploaded_img,
                    model_path=model_path,
                )
        except FileNotFoundError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Lỗi khi load model hoặc dự đoán: {e}")
            return

        # ----- Hiển thị kết quả -----
        st.success(f"✅ Dự đoán: **{pred_label}**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ảnh sau khi resize (64×256, grayscale):**")
            st.image(resized_img, width=256, clamp=True)

        with col2:
            st.markdown("**Xác suất từng lớp:**")
            for cls, p in zip(CLASS_NAMES, probs):
                st.write(f"- `{cls}`: {p:.4f}")

            # Nếu bạn muốn bar chart:
            try:
                import pandas as pd
                df = pd.DataFrame(
                    {"class": CLASS_NAMES, "prob": probs}
                ).set_index("class")
                st.bar_chart(df)
            except ImportError:
                st.info("Cài thêm pandas nếu muốn xem bar chart đẹp hơn.")


if __name__ == "__main__":
    main()
