import json

def combine_texts(extracted_texts):
    combined_text = ""
    for item in extracted_texts:
        page_text = item.get("text", "")
        combined_text += page_text + "\n\n"  # Combine with spacing between pages
    return combined_text

def save_output(text, output_format="plain-text", output_path="output_text"):
    if output_format == "plain-text":
        with open(f"{output_path}.txt", "w", encoding="utf-8") as f:
            f.write(text)
    elif output_format == "json":
        with open(f"{output_path}.json", "w", encoding="utf-8") as f:
            json.dump({"combined_text": text}, f, ensure_ascii=False, indent=4)

def postprocess_texts(extracted_texts, output_format="plain-text", output_path="output_text"):
    combined_text = combine_texts(extracted_texts)
    save_output(combined_text, output_format, output_path)
    return combined_text
