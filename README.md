# 📝 TrOCR: Reconocimiento Óptico de Texto con Transformers

## 1. Resumen (Abstract)
Este proyecto implementa y evalúa el modelo TrOCR (Transformer-based Optical Character Recognition) para el reconocimiento óptico de texto a partir de imágenes. Se utilizaron tanto la implementación original basada en fairseq como la versión moderna en Hugging Face Transformers.

Se realizaron experimentos sobre imágenes con texto impreso y manuscrito, evidenciando que el modelo presenta alto desempeño en texto impreso, pero limitaciones en escritura manuscrita compleja. Adicionalmente, se desarrolló una aplicación interactiva en Streamlit para demostrar el funcionamiento del modelo en tiempo real, permitiendo observar el proceso de inferencia y comparar diferentes configuraciones.

## 2. Introducción
El presente trabajo se basa en el artículo:
- **Título:** TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models
- **Enlace al paper:**[https://arxiv.org/abs/2109.10282](https://arxiv.org/abs/2109.10282)
- **Repositorio original:** [https://github.com/microsoft/unilm/tree/master/trocr](https://github.com/microsoft/unilm/tree/master/trocr)

El reconocimiento óptico de caracteres (OCR) es una tarea fundamental en visión por computador, utilizada en digitalización de documentos, procesamiento de recibos y automatización industrial. Los métodos tradicionales requieren múltiples etapas (detección, segmentación y reconocimiento). TrOCR propone un enfoque *end-to-end* basado puramente en Transformers, eliminando la necesidad de pipelines complejos basados en redes CNN.

El objetivo de este proyecto es:
- Implementar el modelo TrOCR para inferencia.
- Evaluar su desempeño en diferentes tipos de datos secuenciales visuales (texto).
- Analizar su arquitectura interna y limitaciones.
- Construir una demo interactiva funcional con visualización de resultados.

## 3. Marco teórico

### 🔹 Arquitectura Transformer
TrOCR se basa en una arquitectura *encoder-decoder* Transformer estándar:
- **Encoder visual (Vision Transformer - ViT/DeiT/BEiT):** Procesa la imagen dividiéndola en parches y extrae representaciones visuales.
- **Decoder textual (Transformer autoregresivo - RoBERTa/MiniLM):** Genera texto token por token, similar a modelos de lenguaje como GPT.

### 🔹 Innovaciones de TrOCR
- Es el primer OCR completamente *end-to-end* que no utiliza CNNs como *backbone*.
- Hace uso de preentrenamiento masivo uniendo modelos de visión (CV) y lenguaje (NLP).
- Elimina la necesidad de segmentación manual de caracteres o el uso de decodificadores CTC complejos.

### 🧠 ¿Cómo "lee" el modelo? El Mecanismo de Atención y los Tensores Q, K, V
A diferencia de los sistemas OCR tradicionales que "escanean" la imagen de izquierda a derecha de forma rígida, TrOCR utiliza el **Mecanismo de Atención** para "concentrarse" dinámicamente en diferentes partes de la imagen mientras escribe. 

Este proceso se divide en tres fases principales, dominadas por la interacción de tres tensores matemáticos: **Query (Q - Consulta)**, **Key (K - Clave)** y **Value (V - Valor)**. Como se observa en el diagrama, estos tensores se generan multiplicando los embeddings de entrada por matrices de pesos aprendidas ($W_q, W_k, W_v$).

#### 1. El Encoder Visual: Entendiendo la imagen (Self-Attention)
El modelo no procesa la imagen entera de golpe. Primero, la redimensiona a 384x384 píxeles y la **corta en una cuadrícula de parches de 16x16 píxeles**. 

![vectorización](assets/vector.gif)

Dentro del **Encoder (ViT)**, los parches se aplican Auto-Atención (*Self-Attention*) entre sí. 
* **Q, K, V provienen de la misma imagen:** Un parche lanza una consulta (**Q**) a los demás parches (**K**) para buscar el contexto del resto de la palabra y extraer su información visual (**V**). Mediante la fórmula matemática de la atención $\text{softmax}(QK^T/\sqrt{d_k})V$, el modelo fusiona esta información no local.

#### 2. El Decoder Textual: Entendiendo la gramática (Masked Self-Attention)
Mientras el modelo genera el texto, necesita saber qué ha escrito antes para que la palabra tenga sentido. 
* **Q, K, V provienen del texto generado:** Si el modelo ya escribió las letras "L", "I", "C", "E", "N", la nueva consulta (**Q**) analiza las claves (**K**) de esas letras anteriores para deducir el valor (**V**) lógico: que la siguiente letra probablemente sea una "S" para formar "LICENSEE".

#### 3. El Efecto de Concentración: Uniendo Visión y Lenguaje (Cross-Attention)
Aquí ocurre la verdadera magia del OCR. Es el puente de comunicación entre el texto y la imagen (ilustrado en la sección derecha del diagrama inferior).

Para predecir el siguiente carácter, el modelo hace lo siguiente:
1. **Query (Q):** El decoder (texto) dice: *"Ya escribí 'LICEN', ¿qué sigue visualmente?"*
2. **Key (K):** El encoder (imagen) tiene etiquetados todos sus parches visuales y responde: *"Aquí están las coordenadas de la representación latente visual"*.
3. **Value (V):** El modelo **se concentra (Atención Focalizada)** específicamente en el parche de la imagen original donde está dibujada la letra correspondiente, extrae sus píxeles y el decoder finalmente predice la sub-palabra **"SEE"**.

![Diagrama de Arquitectura TrOCR](assets/focus.png)

Este comportamiento explica por qué el modelo es tan potente con texto impreso, pero a veces "alucina" en el manuscrito: si el modelo no logra concentrarse (hacer *match* entre Q y K) por una caligrafía ilegible, el Decoder ignora la imagen y simplemente adivina la siguiente palabra basándose en su modelo de lenguaje interno.

## 4. Metodología
🔧 Entorno
- Python 3.10 (compatibilidad garantizada con librerías base)
- PyTorch + CUDA (aceleración por GPU)
- Hugging Face Transformers (simplificación de inferencia)
- Streamlit (para la interfaz visual interactiva)

🔹 Implementaciones utilizadas
Se trabajó con dos enfoques:
1. **Implementación original (fairseq):** Carga de modelos `.pt` y pipeline manual de preprocesamiento, ideal para entender la arquitectura base a bajo nivel.
2. **Implementación moderna (Hugging Face):** Uso de `TrOCRProcessor` y `VisionEncoderDecoderModel`, pipeline simplificado y robusto para producción y demos.

🔹 Pesos preentrenados
No se entrenó el modelo desde cero. Se cargaron los siguientes pesos:
- `microsoft/trocr-base-printed`
- `microsoft/trocr-base-handwritten`
- `microsoft/trocr-large-handwritten`

## 5. Desarrollo e implementación

▶️ Pasos para ejecutar el proyecto
```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Activar entorno virtual (Ejemplo en Windows/PowerShell)
.\trocr\.venv310\Scripts\Activate.ps1

# 3. Ejecutar la inferencia visual interactiva
streamlit run trocr_demo_app.py
```

🔹 Carga de pesos preentrenados
```python
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
```

🔹 Preprocesamiento y cambio de dimensiones
```python
# El procesador redimensiona la imagen y la convierte en un Tensor de PyTorch.
# Entrada: Imagen JPG/PNG de tamaño variable.
# Salida: Tensor de dimensiones [Batch, Canales, Alto, Ancho] ->[1, 3, 384, 384]
pixel_values = processor(image, return_tensors="pt").pixel_values
```

🔹 Proceso de Inferencia
```python
# El modelo genera la secuencia de tokens de manera autoregresiva (Beam Search).
# Entra el tensor visual y salen IDs de texto. Dimensión de salida: [1, longitud_secuencia]
generated_ids = model.generate(pixel_values, max_new_tokens=50)
```

🔹 Decodificación
```python
# Convierte los IDs numéricos (ej: [0, 234, 54, 2]) a texto legible (sub-palabras).
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## 6. Resultados y análisis
📊 Resultados cualitativos experimentales

| Tipo de texto | Modelo utilizado | Resultado de inferencia |
| :--- | :--- | :--- |
| Impreso (Recibos/Docs) | `trocr-base-printed` | ✔️ Correcto / Muy preciso |
| Manuscrito simple | `trocr-base-handwritten`| ⚠️ Parcial / Errores menores |
| Manuscrito complejo | `trocr-large-handwritten`| ❌ Incorrecto / Alucinación de lenguaje |

📏 Métricas de Desempeño
Para evaluar formalmente el modelo (tal como se indica en el artículo original), se analizan dos métricas principales:
- **CER (Character Error Rate):** Mide la tasa de error a nivel de carácter. En nuestras pruebas cualitativas con texto impreso, el CER tiende a 0%, pero aumenta drásticamente en caligrafía manuscrita compleja.
- **Word Accuracy:** Precisión a nivel de palabra. En imágenes con ruido o escritura cursiva extrema, la precisión cae debido a la mala alineación entre la imagen y el modelo de lenguaje durante la atención cruzada.

⚠️ Comportamiento observado (Limitación Autoregresiva)
Cuando el modelo falla, suele generar texto gramaticalmente coherente pero incorrecto respecto a la imagen. 
- *Ejemplo real obtenido:*
  - **Entrada visual:** Texto manuscrito complejo e ilegible.
  - **Salida del modelo:** *"Prime President of the South of"*
- *Análisis:* Esto evidencia cómo el Decoder (NLP) toma el control absoluto de la predicción cuando el Encoder visual no logra extraer características útiles, "adivinando" texto basándose en su preentrenamiento de lenguaje natural.

## 🖼️ Evidencia Experimental
***Dashboard interactivo de TrOCR (Streamlit)***
![Screenshot Streamlit](assets/TrOCR_screenshot.png)

## 7. Conclusiones
🎯 Aprendizajes
- Es posible realizar OCR de alta calidad de forma completamente *end-to-end* sin requerir redes convolucionales ni lógicas de recorte de caracteres (CTC).
- El desempeño del modelo es altamente sensible al dominio de sus datos de entrenamiento (impreso vs. manuscrito).
- La arquitectura Transformer permite un manejo de contexto gramatical superior al OCR tradicional, pero hereda problemas como la "alucinación" de texto.

⚠️ Limitaciones
- **Falta de detección espacial:** TrOCR requiere imágenes recortadas a nivel de línea de texto (text-line level). No puede procesar una página entera con múltiples bloques de párrafos por sí solo.
- **Alto costo computacional:** Las versiones `Large` y `Base` son pesadas y lentas para procesar masivamente sin aceleración de hardware (GPU).

🚀 Posibles Mejoras
- **Integración de un detector de texto:** Mejorar el pipeline creando una arquitectura de dos etapas, usando un modelo de detección (como CRAFT o DBNet) previo a TrOCR para extraer las cajas delimitadoras de un documento completo.
- **Fine-Tuning local:** Entrenar (ajustar) los pesos del modelo con un dataset propio de caligrafía o tipos de documentos específicos para reducir drásticamente el *Character Error Rate* (CER).
- **Cuantización de Modelos:** Aplicar técnicas de reducción de precisión (ej. de FP32 a INT8) en el modelo `TrOCR-Small` para acelerar el tiempo de inferencia en dispositivos de bajos recursos.

## 8. Referencias
[1] M. Li, T. Lv, L. Cui, Y. Lu, D. Florencio, C. Gu, J. Wang, Z. Zhang, and F. Wei, "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," *arXiv preprint arXiv:2109.10282*, 2021. [En línea]. Disponible en: https://arxiv.org/abs/2109.10282

[2] Microsoft, "TrOCR Original Repository," *GitHub*, 2021. [En línea]. Disponible en: https://github.com/microsoft/unilm/tree/master/trocr

[3] Hugging Face, "TrOCR Documentation and Models," *Hugging Face*, 2023. [En línea]. Disponible en: https://huggingface.co/docs/transformers/model_doc/trocr
