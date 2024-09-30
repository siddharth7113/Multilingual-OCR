import os
from PIL import Image, ImageDraw
from text_extraction import extract_text_english, extract_text_hindi, extract_text_multilingual
from pdf2image import convert_from_path
from docx import Document
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch


def load_model():
    """Load the Qwen2VL model and processor only when a GPU is available."""
    if torch.cuda.is_available():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        print("Model loaded on GPU.")
        return model, processor
    else:
        print("GPU not available. Model loading skipped.")
        return None, None


def image_conversion(file_path, output_folder, max_pages=5):
    """Convert PDF pages to images and save them."""
    images = convert_from_path(file_path, first_page=1, last_page=max_pages)
    image_paths = []
    for i, image in enumerate(images):
        output_image_path = f"{output_folder}/page_{i+1}.png"
        image.save(output_image_path, 'PNG')
        image_paths.append(output_image_path)
    return image_paths

def docx_to_images(docx_path, output_folder):
    """Convert DOCX paragraphs to images and save them."""
    document = Document(docx_path)
    image_files = []
    for i, paragraph in enumerate(document.paragraphs):
        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        img_draw = ImageDraw.Draw(img)
        img_draw.text((10, 10), paragraph.text, fill=(0, 0, 0))
        image_file = os.path.join(output_folder, f'page_{i+1}.png')
        img.save(image_file, 'PNG')
        image_files.append(image_file)
    return image_files

def process_document(file_path, extraction_mode="english"):
    """Process the document using output_images directory."""
     # Load the model and processor
    model, processor = load_model()
    
    # Check if the model is loaded successfully
    if model is None or processor is None:
        raise RuntimeError("Model not loaded because GPU is not available.")

    output_folder = 'output_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if file_path.endswith('.pdf'):
        image_files = image_conversion(file_path, output_folder)
    elif file_path.endswith('.docx'):
        image_files = docx_to_images(file_path, output_folder)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_files = [file_path]
    else:
        raise ValueError("Unsupported file type. Supported formats: PDF, DOCX, PNG, JPG.")

    extracted_texts = []
    for image_file in image_files:
        image = Image.open(image_file)
        if extraction_mode == "english":
            extracted_text = extract_text_english(image, processor, model)
        elif extraction_mode == "hindi":
            extracted_text = extract_text_hindi(image, processor, model)
        elif extraction_mode == "multilingual":
            extracted_text = extract_text_multilingual(image, processor, model)
        else:
            raise ValueError("Unsupported extraction mode. Choose from 'english', 'hindi', or 'multilingual'.")

        extracted_texts.append({"page": image_file, "text": extracted_text})

    return extracted_texts
