import os
import shutil
from PIL import Image
from violence_detection import filter_random
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialiser le processeur et le modèle Blip
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def filter_and_generate_captions(input_folder_frames, output_folder, output_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with open(output_file, "w") as file:
            frame_count = 0
            for filename in os.listdir(input_folder_frames):
                if filename.endswith(".jpg"):
                    frame_path = os.path.join(input_folder_frames, filename)

                    if filter_random(frame_count):
                        shutil.copy(frame_path, output_folder)

                        raw_image = Image.open(frame_path).convert('RGB')
                        inputs = processor(raw_image, return_tensors="pt")
                        out = model.generate(**inputs)
                        generated_caption = processor.decode(out[0], skip_special_tokens=True)

                        file.write(generated_caption + "\n")

                    frame_count += 1
    except Exception as e:
        print("Une erreur s'est produite lors de la génération des légendes :", e)