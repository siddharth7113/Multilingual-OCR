import streamlit as st
from PIL import Image
import os
from main import process_document
from postprocessing import postprocess_texts
from qa_search import basic_ocr_search, advanced_qa_search, load_models
import time
import json

# Utility functions for file-based state management
def save_state(file_name, data):
    """Safely save the OCR state to a file."""
    tmp_file = file_name + ".tmp"
    with open(tmp_file, 'w') as file:
        json.dump(data, file)
    os.rename(tmp_file, file_name)

def load_state(file_name):
    """Load the OCR state from a file."""
    if os.path.exists(file_name):
        try:
            with open(file_name, 'r') as file:
                file_content = file.read().strip()
                if file_content:
                    return json.loads(file_content)
                else:
                    print("Warning: The file is empty. Initializing default state.")
                    return {"combined_text": "", "file_path": "", "extracted_texts": None}  # Return a default state
        except json.JSONDecodeError:
            print("Error: Corrupted JSON. Reinitializing state.")
            return {"combined_text": "", "file_path": "", "extracted_texts": None}  # Return a default state
    return {"combined_text": "", "file_path": "", "extracted_texts": None}  # File doesn't exist, return default


# Custom CSS to style the app
st.markdown("""
    <style>
    .title {
        font-size: 2.5em;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .output-box {
        padding: 10px;
        border: 2px solid #ddd;
        border-radius: 10px;
         height: 400px;
        background-color: #fff;
    }
    .search-box {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 8px;
    }
    .sidebar .sidebar-content {
        width: 350px;
    }
    </style>
    <h1 class="title">📄 Multilingual OCR Conversion 🌍</h1>
""", unsafe_allow_html=True)

# Sidebar: Document Upload, Preview, and Language Selection
st.sidebar.header("🗂️ Upload your document")
uploaded_file = st.sidebar.file_uploader("Drag and drop or browse file", type=["pdf", "docx", "png", "jpg", "jpeg"])

# Preview for image files only (in sidebar)
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension in ['png', 'jpg', 'jpeg']:
        st.sidebar.image(Image.open(uploaded_file), caption="🖼️ Uploaded Image Preview", use_column_width=True)
    else:
        st.sidebar.warning("Preview is only available for image files (PNG, JPG, JPEG).")

# Load saved state if exists
ocr_state = load_state("ocr_state.json") or {"combined_text": "", "file_path": "", "extracted_texts": None}

# Add a "Clear" button to reset the OCR text and state
if st.button("🧹 Clear", key="clear_button", help="Clear OCR Text and Uploaded File"):
    ocr_state = {"combined_text": "", "file_path": "", "extracted_texts": None}
    save_state("ocr_state.json", ocr_state)
    st.experimental_rerun()  # Force refresh to clear the interface

# Language selection in sidebar
language = st.sidebar.radio("🌐 Select Language for Extraction", ('📖 English', '🇮🇳 Hindi', '🌍 Multilingual'))

# "Start OCR" button
start_ocr = st.sidebar.button("🚀 Start OCR Processing")

# Search Type selection
search_type = st.sidebar.radio("Search Type", ('🔍 Basic Search', '💡 Advanced QA Search'))

# Perform OCR and Display Results
if start_ocr and uploaded_file is not None:
    # Ensure the output directory exists
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the uploaded file to the output directory
    file_path = os.path.join(output_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform OCR (Simulate time delay for progress)
    st.text("Processing document, please wait...")
    time.sleep(2)  # Simulate processing time
    extracted_texts = process_document(file_path, extraction_mode=language.split(' ')[1].lower())

    # Post-process the extracted text
    combined_text = postprocess_texts(extracted_texts)

    # Save to file-based state
    ocr_state = {
        "combined_text": combined_text,
        "file_path": file_path,
        "extracted_texts": extracted_texts
    }
    save_state("ocr_state.json", ocr_state)

# Main Screen: OCR Text Output and Search Functionality
st.subheader("📝 OCR Text Output")

# Only show the OCR text area if there's actual content
if ocr_state['combined_text']:
    st.markdown('<div class="output-box">', unsafe_allow_html=True)
    st.text_area("Extracted OCR Text", value=ocr_state['combined_text'], height=300)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No OCR text to display. Please upload a document and start the OCR process.")

# Search functionality
if search_type == '🔍 Basic Search':
    st.markdown('<h3>🔍 Basic Search in OCR Text</h3>', unsafe_allow_html=True)
    search_query = st.text_input("Search for specific words/phrases in the extracted text", value="")

    # Search Type Options
    search_mode = st.selectbox("Select Search Mode", ["Exact", "Fuzzy", "Case-Insensitive", "Wildcard", "Regex"])

    # Perform the basic search when search button is clicked
    if st.button("🔍 Search Text") and ocr_state['combined_text']:
        search_results = basic_ocr_search(ocr_state['extracted_texts'], search_query, search_type=search_mode.lower())
        st.write(search_results if search_results else "No results found.")

elif search_type == '💡 Advanced QA Search':
    st.markdown('<h3>💡 Advanced QA Search</h3>', unsafe_allow_html=True)
    qa_query = st.text_input("Ask a question about the document", value="")

    # Load models if a question is asked
    if st.button("💡 Run QA Search") and qa_query and ocr_state['file_path']:
        # Load the QA models
        RAG, qwen_model, qwen_processor = load_models()

        # Perform advanced QA search
        answer = advanced_qa_search(ocr_state['file_path'], qa_query, RAG, qwen_model, qwen_processor)
        st.write(answer)

# Download Options
if ocr_state['combined_text']:
    st.subheader("💾 Download Options")

    output_format = st.radio("Select Download Format", ('📄 Plain-Text', '🗃️ JSON'))

    if output_format == "📄 Plain-Text":
        st.download_button(
            label="💾 Download as Text",
            data=ocr_state['combined_text'],
            file_name="extracted_text.txt"
        )
    elif output_format == "🗃️ JSON":
        json_data = {"combined_text": ocr_state['combined_text']}
        st.download_button(
            label="💾 Download as JSON",
            data=json.dumps(json_data, ensure_ascii=False, indent=4),
            file_name="extracted_text.json"
        )

# Footer
st.markdown('<p class="footer">Built with ❤️ using Streamlit</p>', unsafe_allow_html=True)
