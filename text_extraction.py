from qwen_vl_utils import process_vision_info

def extract_text_english(image, processor, model, text_query="Provide all the English text present in the image. Do not introduce additional details, assumptions, or biases."):
    messages = [
        {"role":"user",
         "content":[{"type":"image", "image":image},
                    {"type":"text","text":text_query}]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    generate_ids = model.generate(**inputs, max_new_tokens=2000)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0]


def extract_text_hindi(image, processor, model, text_query="Provide all the Hindi text present in the image. Do not introduce additional details, assumptions, or biases."):
    messages = [
        {"role":"user",
         "content":[{"type":"image", "image":image},
                    {"type":"text","text":text_query}]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    generate_ids = model.generate(**inputs, max_new_tokens=2000)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0]

def extract_text_multilingual(image, processor, model, text_query="Provide all the text present in the image in English and Hindi languages. Do not introduce additional details, assumptions, or biases."):
    messages = [
        {"role":"user",
         "content":[{"type":"image", "image":image},
                    {"type":"text","text":text_query}]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    generate_ids = model.generate(**inputs, max_new_tokens=500)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return output_text[0]
