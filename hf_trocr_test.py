from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# CAMBIA esta ruta por una imagen tuya con texto impreso claro
IMAGE_PATH = r"C:\Github\unilm\imagen.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

image = Image.open(IMAGE_PATH).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

generated_ids = model.generate(pixel_values, max_new_tokens=50)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Texto reconocido:", generated_text)