import gradio as gr
from ultralytics import YOLO

model = YOLO("best.pt")

def iceri_onerisi(cilt_tipi):
    oneriler = {
        "kuru cilt": [
            "Hyalüronik Asit", "Gliserin", "Seramidler",
            "Shea Butter, Skualen", "Panthenol (B5)",
            "Aloe Vera", "Düşük oranlı Laktik Asit"
        ],
        "yagli cilt": [
            "Salicylic Acid (BHA)", "Niacinamide (B3)", "Çinko (Zinc PCA)",
            "Retinoidler (Retinol)", "Yeşil çay özü, Centella Asiatica",
            "Kil maskesi (haftada 1-2)"
        ],
        "karma cilt": [
            "Hafif su bazlı nemlendiriciler", "T bölgesine BHA",
            "Yanlara Laktik Asit veya PHA", "Vitamin C ve E", "Centella Asiatica"
        ],
        "siyah nokta": [
            "Salicylic Acid (BHA)", "Retinoidler (Retinol)", "Kil maskesi",
            "Günlük nazik temizleyici", "Gözenek sıkılaştırıcı ürünler"
        ]
    }
    return "\n".join(f"- {x}" for x in oneriler.get(cilt_tipi, []))

def predict(img):
    results = model.predict(img, verbose=False)

    if len(results[0].boxes) > 0:
        cls_ids = [int(cls) for cls in results[0].boxes.cls]
        cilt_tipleri = ["karma cilt", "kuru cilt", "siyah nokta", "yagli cilt"]
        tahminler = list(set(cilt_tipleri[cls_id] for cls_id in cls_ids))
    else:
        return "Cilt tipi tespit edilemedi", "Bu cilt tipi için öneri bulunamadı."

    oneriler_listesi = []
    for tahmin in tahminler:
        oneriler_listesi.extend(iceri_onerisi(tahmin).split('\n'))
    oneriler_listesi = list(dict.fromkeys([o.strip('- ').strip() for o in oneriler_listesi if o]))
    oneriler_text = "\n".join(f"- {o}" for o in oneriler_listesi)
    tahmin_str = ", ".join(tahminler).capitalize()

    print(f"Tespit edilen cilt tipleri: {tahmin_str}")
    print(f"Öneriler:\n{oneriler_text}")

    return tahmin_str, oneriler_text

with gr.Blocks() as demo:
    gr.Markdown("## Cilt Tipi Analizi ve İçerik Önerisi (YOLOv8)")
    gr.Markdown("Webcam veya dosya yükleyerek cilt tipinizi tespit edin.")

    with gr.Row():
        image_input = gr.Image(
            sources=["upload", "webcam"],
            type="numpy",
            label="Görsel Seç veya Kameradan Çek"
        )
        cilt_output = gr.Textbox(label="Tahmin Edilen Cilt Tipi")
        öneri_output = gr.Textbox(label="Önerilen İçerikler", lines=10)

    btn = gr.Button("Analiz Et")
    btn.click(fn=predict, inputs=image_input, outputs=[cilt_output, öneri_output])

demo.launch()
