import streamlit as st
from PIL import Image
import os
from main import process_document
from postprocessing import postprocess_texts
from qa_search import basic_ocr_search, advanced_qa_search, load_models
import time
import json

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

# Language selection in sidebar
language = st.sidebar.radio("🌐 Select Language for Extraction", ('📖 English', '🇮🇳 Hindi', '🌍 Multilingual'))

# "Start OCR" button
start_ocr = st.sidebar.button("🚀 Start OCR Processing")

# Search Type selection
search_type = st.sidebar.radio("Search Type", ('🔍 Basic Search', '💡 Advanced QA Search'))

# Placeholder for extracted text and combined_text in session state
combined_text = st.session_state.get('combined_text', "")
file_path = st.session_state.get('file_path', "")

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
    st.session_state['combined_text'] = combined_text
    st.session_state['file_path'] = file_path

# Main Screen: OCR Text Output and Search Functionality
st.subheader("📝 OCR Text Output")

# Show OCR text only if available
if combined_text:
    st.markdown('<div class="output-box">', unsafe_allow_html=True)
    st.text_area("Extracted OCR Text", value=combined_text, height=300)
    st.markdown('</div>', unsafe_allow_html=True)

# Search functionality
if search_type == '🔍 Basic Search':
    st.markdown('<h3>🔍 Basic Search in OCR Text</h3>', unsafe_allow_html=True)
    search_query = st.text_input("Search for specific words/phrases in the extracted text", value="")
    
    # Search Type Options
    search_mode = st.selectbox("Select Search Mode", ["Exact", "Fuzzy", "Case-Insensitive", "Wildcard", "Regex"])

    # Perform the basic search when search button is clicked
    if st.button("🔍 Search Text") and combined_text:
        search_results = basic_ocr_search(extracted_texts, search_query, search_type=search_mode.lower())
        st.write(search_results if search_results else "No results found.")

elif search_type == '💡 Advanced QA Search':
    st.markdown('<h3>💡 Advanced QA Search</h3>', unsafe_allow_html=True)
    qa_query = st.text_input("Ask a question about the document", value="")

    # Load models if a question is asked
    if st.button("💡 Run QA Search") and qa_query and file_path:
        # Load the QA models
        RAG, qwen_model, qwen_processor = load_models()
        
        # Perform advanced QA search
        answer = advanced_qa_search(file_path, qa_query, RAG, qwen_model, qwen_processor)
        st.write(answer)

# Download Options
if combined_text:
    st.subheader("💾 Download Options")
    
    output_format = st.radio("Select Download Format", ('📄 Plain-Text', '🗃️ JSON'))

    if output_format == "📄 Plain-Text":
        st.download_button(
            label="💾 Download as Text",
            data=combined_text,
            file_name="extracted_text.txt"
        )
    elif output_format == "🗃️ JSON":
        json_data = {"combined_text": combined_text}
        st.download_button(
            label="💾 Download as JSON",
            data=json.dumps(json_data, ensure_ascii=False, indent=4),
            file_name="extracted_text.json"
        )

# Footer
st.markdown('<p class="footer">Built with ❤️ using Streamlit</p>', unsafe_allow_html=True)
