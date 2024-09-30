import os
from PIL import Image
from pdf2image import convert_from_path
from qa_search import load_models, advanced_qa_search, basic_ocr_search
from main import process_document

# Convert PDF to PNG images (limit to first 5 pages)
def convert_pdf_to_images(pdf_path, output_folder="output_images", max_pages=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert only the first `max_pages` pages of the PDF to images
    images = convert_from_path(pdf_path, first_page=1, last_page=max_pages)
    image_files = []
    for i, image in enumerate(images):
        output_image_path = os.path.join(output_folder, f"page_{i+1}.png")
        image.save(output_image_path, "PNG")
        image_files.append(output_image_path)
    
    return image_files

# Test the Basic Search (OCR)
def test_basic_search():
    pdf_path = "Banbhatta_Saaransh.pdf"  # Path to the PDF file to convert
    image_files = convert_pdf_to_images(pdf_path)
    
    # Simulate some OCR-extracted text from these image files (in practice, you'd use OCR here)
    extracted_texts = [
        {"page": image_files[0], "text": "This is a test document. It contains important information."},
        {"page": image_files[1], "text": "Second page with some more information. Here is a keyword."},
        {"page": image_files[2], "text": "Final page with the conclusion."}
    ]
    
    search_query = "keyword"
    results = basic_ocr_search(extracted_texts, search_query)
    print("Basic OCR Search Results:")
    print(results)

# Test the QA Search (Advanced)
def test_qa_search():
    # Load models for QA
    qwen_model, qwen_processor = load_models()  # Only two models are returned
    
    # Path to document to index and search (converted PDF images will be used)
    document_path = "Banbhatta_Saaransh.pdf"  # Ensure this PDF exists for testing
    
    # Query for the QA model
    text_query = "What is the language in the document?"
    
    # Perform the advanced QA search
    answer = advanced_qa_search(document_path, text_query, qwen_model, qwen_processor)
    print("Advanced QA Search Results:")
    print(answer)

# Run tests
if __name__ == "__main__":
    print("Running Basic Search Test...")
    test_basic_search()

    print("\nRunning Advanced QA Search Test...")
    test_qa_search()
