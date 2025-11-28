# app.py
import numpy as np
from pathlib import Path

import streamlit as st
from PIL import Image

# ---- Keras (TF) ----
from tensorflow import keras

# ---- PyTorch ----
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ================== CẤU HÌNH ==================

IMG_H, IMG_W = 64, 256
CLASS_NAMES = ["display", "monospace", "san_serif", "script", "serif"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model mặc định dùng cho app (có thể là .keras hoặc .pth)
DEFAULT_MODEL_PATH = "/workspaces/Font-detection/CNN/best_model.keras"
# hoặc: DEFAULT_MODEL_PATH = "/workspaces/Font-detection/ViT/vit_font_best.pth"


# ================== BUILD MODEL TORCH (CẦN SỬA CHO ĐÚNG) ==================

class PatchEmbedding(nn.Module):
    def __init__(self, img_h, img_w, patch_size, in_chans, emb_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # số patch: (H/P) * (W/P)
        num_patches = (img_h // patch_size) * (img_w // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, emb_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, emb_dim]

        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, D]

        x = x + self.pos_embed[:, : N + 1, :]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class SimpleViT(nn.Module):
    def __init__(
        self,
        img_h=64,
        img_w=256,
        patch_size=16,
        in_chans=1,
        emb_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        num_classes=5,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_h, img_w, patch_size, in_chans, emb_dim
        )
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=emb_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)  # [B, N+1, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)         # [B, N+1, D]
        cls_token = x[:, 0]      # [B, D]
        logits = self.head(cls_token)  # [B, num_classes]
        return logits


def build_torch_model():
    """
    ViT đơn giản cho ảnh 1×64×256, 5 lớp.
    Nếu lúc train bạn dùng đúng kiến trúc này,
    vit_font_best.pth sẽ load được thẳng.
    """
    model = SimpleViT(
        img_h=IMG_H,
        img_w=IMG_W,
        patch_size=16,
        in_chans=1,
        emb_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        num_classes=len(CLASS_NAMES),
        dropout=0.1,
    )
    return model


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

def run_font_classifier(image_file):
    """
    - Đọc ảnh
    - Load model từ DEFAULT_MODEL_PATH (Keras hoặc Torch)
    - Trả về: (pred_label, probs, resized_img, backend)
    """
    pil_img = Image.open(image_file)

    model_info = load_font_model(DEFAULT_MODEL_PATH)
    backend = model_info["backend"]
    model = model_info["model"]

    if backend == "keras":
        x, resized_img = preprocess_for_keras(pil_img)
        probs = model.predict(x)[0]
    elif backend == "torch":
        x, resized_img = preprocess_for_torch(pil_img)
        with torch.no_grad():
            logits = model(x)
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

    # ----- Upload ảnh -----
    uploaded_img = st.file_uploader(
        "Tải lên ảnh font (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_img is not None:
        st.subheader("Ảnh gốc")
        st.image(uploaded_img, use_container_width=True)

    if st.button("🚀 Dự đoán font family"):
        if uploaded_img is None:
            st.error("Bạn cần tải lên một ảnh trước.")
            return

        try:
            with st.spinner("Đang phân loại font..."):
                pred_label, probs, resized_img, backend = run_font_classifier(
                    uploaded_img
                )
        except (FileNotFoundError, ValueError) as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Lỗi khi load model hoặc dự đoán: {e}")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Ảnh sau khi resize:**")
            st.image(resized_img, width=256, clamp=True)

        with col2:
            st.markdown("**Xác suất từng lớp:**")
            for cls, p in zip(CLASS_NAMES, probs):
                st.write(f"- `{cls}`: {p:.4f}")


if __name__ == "__main__":
    main()
