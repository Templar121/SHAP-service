from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import numpy as np
from PIL import Image
import shap, tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
label_mapping = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}
class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
MODEL_PATH = Path(__file__).parent / "model" / "best_model.keras"
TEMP_DIR = Path(__file__).parent / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# --- Load Model ---
model = load_model(str(MODEL_PATH), compile=False)

# --- Utility Functions ---
def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def preprocess_image_from_array(img_array):
    img = tf.image.resize(img_array, (64, 64)).numpy().astype(np.float32)
    return np.expand_dims(img, axis=0)

def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.get("/health")
def health_check():
    return {"status": "API is healthy ✅"}

@app.post("/explain")
async def explain_with_shap(file: UploadFile = File(...)):
    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        # Save uploaded file to temp directory
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        arr = np.squeeze(preprocess_image(temp_path))  # shape: (64, 64, 3)
        preds = model.predict(preprocess_image(temp_path), verbose=0)[0]

        masker = shap.maskers.Image("inpaint_telea", arr.shape)
        def shap_predict(x):
            batch = np.stack([preprocess_image_from_array(im)[0] for im in x])
            return model.predict(batch, verbose=0)

        explainer = shap.Explainer(shap_predict, masker, output_names=class_names)
        explanation = explainer(arr[np.newaxis, ...], max_evals=50)
        shap_vals = explanation.values[0]

        top_cls = np.argsort(preds)[::-1][:2]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, cls in zip(axes, top_cls):
            overlay = shap_vals[:, :, :, cls].mean(axis=-1)
            overlay -= overlay.mean()
            overlay /= (np.abs(overlay).max() + 1e-8)
            ax.imshow(arr.astype(np.uint8))
            ax.imshow(overlay, cmap='bwr', alpha=0.5, vmin=-1, vmax=1)
            ax.set_title(f"{class_names[cls]} ({preds[cls]:.2f})")
            ax.axis('off')

        return {"shap_base64": fig_to_base64(fig)}

    except Exception as e:
        return {"error": str(e)}

    finally:
        try:
            temp_path.unlink()  # Clean up temp file
        except Exception:
            pass
