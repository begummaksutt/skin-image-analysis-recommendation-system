import os
import numpy as np
import cv2
import gradio as gr
from ultralytics import YOLO

# =======================
# MODEL YOLU
# =======================
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} bulunamadı. Dosya yolunu kontrol et!")

# Modeli yükle
model = YOLO(MODEL_PATH)

# =======================
# PREDICT FONKSİYONU
# =======================
def predict(image):
    # 1. Gelen görüntüyü kaydet (kontrol amaçlı)
    img_np = np.array(image)
    cv2.imwrite("last_input.jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # 2. YOLO ile tahmin yap
    results = model(image)

    # 3. Tahminli çıktıyı döndür
    return results[0].plot()

# =======================
# GRADIO ARAYÜZÜ
# =======================
app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Resim yükle veya kamera kullan (alttan seç)")
    ],
    outputs=gr.Image(type="pil", label="Tahmin Sonucu"),
    live=True,
    title="Cilt Analizi - YOLOv8",
    description="best.pt modeliyle nesne tespiti/cilt analizi"
)

if __name__== "__main__":
    app.launch()