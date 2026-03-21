import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(
    page_title="Demo TrOCR",
    page_icon="📝",
    layout="wide"
)

@st.cache_resource
def load_model(model_name: str):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    return processor, model

st.title("📝 Demo TrOCR: OCR con Transformers")
st.markdown(
    """
    Esta aplicación demuestra el funcionamiento de **TrOCR** para reconocimiento óptico de texto.
    Permite comparar el comportamiento de un modelo especializado en **texto impreso**
    y otro especializado en **texto manuscrito**.
    """
)

with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader(
        "Carga una imagen",
        type=["png", "jpg", "jpeg"]
    )

    model_option = st.selectbox(
        "Selecciona el tipo de modelo",
        options=[
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-base-printed"
        ],
        index=0
    )

    run_button = st.button("Ejecutar OCR")

device = "cuda" if torch.cuda.is_available() else "cpu"

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Imagen de entrada")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagen cargada", width="stretch")
        
        # Alerta sobre multilínea si la imagen es muy alta
        w, h = image.size
        if h > w * 0.5:
             st.warning("⚠️ **Nota:** Esta imagen parece tener varias líneas o un formato vertical. TrOCR funciona mejor con imágenes que contienen **una sola línea** de texto.")
    else:
        st.info("Carga una imagen para comenzar.")

with col2:
    st.subheader("Resultado OCR")

    if uploaded_file is not None and run_button:
        with st.spinner("Cargando modelo y ejecutando OCR..."):
            processor, model = load_model(model_option)
            model = model.to(device)

            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=50,
                num_beams=5,
                early_stopping=True
            )
            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

        st.success("OCR ejecutado correctamente")
        st.markdown(f"**Modelo usado:** `{model_option}`")
        st.markdown(f"**Dispositivo:** `{device}`")
        st.text_area("Texto reconocido", generated_text, height=150)
    elif uploaded_file is not None:
        st.info("Presiona **Ejecutar OCR** para procesar la imagen.")
    else:
        st.info("Aquí aparecerá el texto reconocido.")

st.markdown("---")
st.subheader("Arquitectura del modelo")
st.markdown(
    """
    **TrOCR** sigue un esquema de tipo **encoder-decoder**:

    - **Encoder visual:** recibe la imagen y extrae representaciones visuales.
    - **Decoder autoregresivo:** genera el texto token por token.
    - **Salida final:** secuencia textual reconocida desde la imagen.

    **Notas Técnicas Relevantes:**
    - **Aviso "MISSING encoder.pooler.dense"**: Es normal. TrOCR no utiliza la capa de *pooling* del encoder ViT (diseñada para clasificación de imágenes) porque el decoder necesita la secuencia completa de parches. Puedes ignorar este mensaje.
    - **Segmentación por líneas**: TrOCR es un modelo de **nivel de línea**. Para documentos completos, se requiere un paso previo de segmentación de líneas.
    - **Modelos**:
        - El modelo **printed** es ideal para formularios y documentos digitales.
        - El modelo **handwritten** sobresale en texto escrito a mano (entrenado con el dataset IAM).
    """
)