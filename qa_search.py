# qa_search.py
import os
from main import image_conversion, docx_to_images, load_model  # Reusing from main.py
from text_extraction import extract_text_english, extract_text_hindi, extract_text_multilingual
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw
from fuzzywuzzy import fuzz

# Load the model and processor for QA
def load_models():
    """Load the Qwen2VL model and the RAG model for QA."""
    model, processor = load_model()
    return model, processor

# Perform advanced QA search
def advanced_qa_search(document_path, text_query, model, processor, max_pages=5):
    """
    Perform a QA-based search by first converting the document to images,
    then running the QA model on the extracted information.
    """
    # Convert document (PDF or DOCX) to images, limited to the first `max_pages` pages
    output_folder = 'output_images'
    if document_path.endswith('.pdf'):
        image_files = image_conversion(document_path, output_folder, max_pages=max_pages)
    elif document_path.endswith('.docx'):
        image_files = docx_to_images(document_path, output_folder)
    elif document_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_files = [document_path]
    else:
        raise ValueError("Unsupported file type. Supported formats: PDF, DOCX, PNG, JPG.")
    
    # Perform the QA-based search using the images and the provided text query
    extracted_texts = []
    for image_file in image_files:
        image = Image.open(image_file)
        messages = [
            {"role": "user", 
             "content": [{"type": "image", "image": image}, {"type": "text", "text": text_query}]}
        ]
        
        # Process the vision info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text_query], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        # Generate the answer from the model
        generate_ids = model.generate(**inputs, max_new_tokens=1000)
        output_text = processor.batch_decode(generate_ids, skip_special_tokens=True)
        
        extracted_texts.append({"page": image_file, "text": output_text[0]})

    return extracted_texts

def basic_ocr_search(extracted_texts, search_query, search_type="exact"):
    results = []

    # Loop through each page's text
    for entry in extracted_texts:
        page_text = entry.get("text", "")
        page_number = entry.get("page", "")

        # Exact Match
        if search_type == "exact" and search_query.lower() in page_text.lower():
            results.append(f"Page {page_number}: {page_text}")
        
        # Fuzzy Match
        elif search_type == "fuzzy":
            if fuzz.partial_ratio(search_query.lower(), page_text.lower()) > 75:
                results.append(f"Page {page_number}: {page_text}")
        
        # Case-Insensitive Match
        elif search_type == "case-insensitive" and search_query.lower() in page_text.lower():
            results.append(f"Page {page_number}: {page_text}")

        # Wildcard Search
        elif search_type == "wildcard":
            wildcard_pattern = search_query.replace("*", ".*").replace("?", ".")
            if re.search(wildcard_pattern, page_text, re.IGNORECASE):
                results.append(f"Page {page_number}: {page_text}")
        
        # Regex Search
        elif search_type == "regex":
            if re.search(search_query, page_text, re.IGNORECASE):
                results.append(f"Page {page_number}: {page_text}")

    return results if results else ["No results found."]
