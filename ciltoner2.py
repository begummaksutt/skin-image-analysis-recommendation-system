import gradio as gr
from ultralytics import YOLO

# YOLO modelini yükle
model = YOLO("best.pt")

icerik_onerileri = {
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

urun_linkleri = {
    "kuru cilt": [
        "https://www.bepanthol.com.tr/urunlerimiz/gunluk-cilt-bakimi/bepanthol-cilt-bakim-kremi",
        "https://www.bepanthol.com.tr/urunlerimiz/gunluk-cilt-bakimi/bepanthol-derma-onarici-bakim-merhemi"
    ],
    "karma cilt": [
        "https://farmasi.com.tr/farmasi/product-detail/dr-c-tuna-cay-aac-ya-yuz-kremi-50-ml?pid=1000290",
        "https://farmasi.com.tr/farmasi/product-detail/dr-c-tuna-age-reversist-nemlendirici-krem-30-ml?pid=1000271"
    ],
    "yagli cilt": [
        "https://farmasi.com.tr/farmasi/product-detail/dr-c-tuna-aynsefa-ya-yuz-kremi-50-ml?pid=1000285",
        "https://farmasi.com.tr/farmasi/product-detail/dr-c-tuna-aqua-yuz-kremi-50-ml?pid=1000267"
    ],
    "siyah nokta": [
        "https://farmasi.com.tr/farmasi/product-detail/dr-c-tuna-cay-aac-ya-serumu-10-ml?pid=1000292",
        "https://farmasi.com.tr/farmasi/product-detail/dr-c-tuna-cay-aac-ya-yuz-temizleme-tonii-125-ml?pid=1000289"
    ]
}

def predict(img):
    results = model.predict(img, conf=0.10, verbose=False)

    if len(results[0].boxes) > 0:
        cls_ids = [int(cls) for cls in results[0].boxes.cls]
        cilt_tipleri = ["karma cilt", "kuru cilt", "siyah nokta", "yagli cilt"]
        tahminler = list(set(cilt_tipleri[cls_id] for cls_id in cls_ids))

        tahmin_sonucu = ", ".join(tahminler)

        icerik = "\n".join(f"- {x}" for x in icerik_onerileri.get(tahminler[0], []))

        urunler = "<br>".join(f"<a href='{link}' target='_blank'>{link}</a>" for link in urun_linkleri.get(tahminler[0], []))

        return tahmin_sonucu, icerik, urunler
    else:
        return "Cilt tipi tespit edilemedi", "-", "-"

with gr.Blocks() as demo:
    gr.Markdown("## Cilt Tipi Tespiti ve Ürün Önerisi")
    with gr.Row():
        inp = gr.Image(type="filepath", label="Yüz Fotoğrafı Yükle")
        with gr.Column():
            tahmin = gr.Textbox(label="Tespit Edilen Cilt Tipi")
            icerik = gr.Textbox(label="İçerik Önerileri")
            urun = gr.HTML(label="Ürün Linkleri")  # Burayı gr.HTML yaptık

    inp.change(predict, inputs=inp, outputs=[tahmin, icerik, urun])

demo.launch()
