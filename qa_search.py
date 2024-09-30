# qa_search.py
import os
import re
import torch
from main import image_conversion, docx_to_images, load_model  # Reusing from main.py
from text_extraction import extract_text_english, extract_text_hindi, extract_text_multilingual
from qwen_vl_utils import process_vision_info
from PIL import Image
from fuzzywuzzy import fuzz
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from byaldi import RAGMultiModalModel  
import streamlit as st


# Load models (this can be called by the app when needed)
# Cache the model loading to avoid reloading every time
@st.cache_resource
def load_models():
    try:
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16).cuda()
        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        return RAG, qwen_model, qwen_processor
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def advanced_qa_search(document_path, text_query, rag_model, vision_model, processor, max_pages=5):
    try:
        output_folder = 'output_images'
        
        # Convert the document into images based on the file type
        if document_path.endswith('.pdf'):
            image_files = image_conversion(document_path, output_folder, max_pages=max_pages)
        elif document_path.endswith('.docx'):
            image_files = docx_to_images(document_path, output_folder)
        elif document_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files = [document_path]
        else:
            raise ValueError("Unsupported file type. Supported formats: PDF, DOCX, PNG, JPG.")
        
        # Index the document images using Byaldi RAG, with overwrite=True to replace any previous index
        index_name = "image_index"
        rag_model.index(input_path=document_path, index_name=index_name, overwrite=True)

        # Perform the QA-based search using RAG's search method
        results = rag_model.search(text_query, k=1)

        # Extract and process the relevant information using Qwen2-VL model
        extracted_texts = []
        for result in results:
            image_index = result['page_num'] - 1  # Adjust for 0-based index
            image = Image.open(image_files[image_index])
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_query}]}]

            # Process the vision and text information
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text_query], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda")

            # Generate the answer from the Qwen2-VL model
            generate_ids = vision_model.generate(**inputs, max_new_tokens=1000)
            output_text = processor.batch_decode(generate_ids, skip_special_tokens=True)

            # Store the result
            extracted_texts.append({"page": image_files[image_index], "text": output_text[0]})

        return extracted_texts
    finally:
        torch.cuda.empty_cache()  # Free up GPU memory after processing


# Basic OCR search
def basic_ocr_search(extracted_texts, search_query, search_type="exact", snippet_length=100):
    results = []
    
    # Loop through each page's text and perform the selected search type
    for entry in extracted_texts:
        page_text = entry.get("text", "")
        page_number = entry.get("page", "")
        
        # Exact Match
        if search_type == "exact":
            start_index = page_text.lower().find(search_query.lower())
            if start_index != -1:
                snippet = page_text[max(0, start_index - snippet_length):start_index + snippet_length]
                results.append(f"Page {page_number}: ...{snippet}...")
        
        # Fuzzy Match
        elif search_type == "fuzzy":
            if fuzz.partial_ratio(search_query.lower(), page_text.lower()) > 75:
                snippet = page_text[:snippet_length]
                results.append(f"Page {page_number}: ...{snippet}...")
        
        # Case-Insensitive Match
        elif search_type == "case-insensitive":
            start_index = page_text.lower().find(search_query.lower())
            if start_index != -1:
                snippet = page_text[max(0, start_index - snippet_length):start_index + snippet_length]
                results.append(f"Page {page_number}: ...{snippet}...")
        
        # Wildcard Search (converted to regex)
        elif search_type == "wildcard":
            wildcard_pattern = re.escape(search_query).replace(r"\*", ".*").replace(r"\?", ".")
            matches = re.finditer(wildcard_pattern, page_text, re.IGNORECASE)
            for match in matches:
                start_index = match.start()
                snippet = page_text[max(0, start_index - snippet_length):start_index + snippet_length]
                results.append(f"Page {page_number}: ...{snippet}...")
        
        # Regex Search
        elif search_type == "regex":
            matches = re.finditer(search_query, page_text, re.IGNORECASE)
            for match in matches:
                start_index = match.start()
                snippet = page_text[max(0, start_index - snippet_length):start_index + snippet_length]
                results.append(f"Page {page_number}: ...{snippet}...")
    
    return results if results else ["No results found."]
