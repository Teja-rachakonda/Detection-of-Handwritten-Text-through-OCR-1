#Big picture: This app takes a handwritten image → extracts text using OCR (TrOCR + EasyOCR) → cleans it → sends it to a Hugging Face LLM to fix spelling/grammar → 
#lets the user manually edit → downloads final text


import os
import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import requests
from utils import preprocess_image, segment_lines, segment_words, filter_ocr_output
from dotenv import load_dotenv

# Load env variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def hf_llm_correction(text):
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"  # You can use your preferred model
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    prompt = f"You are a helpful assistant. Correct this handwritten OCR text for spelling, grammar, and clarity:\n\n{text}"
    payload = {"inputs": prompt, "options":{"use_cache":False, "wait_for_model":True}}

    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if response.status_code == 200:
        data = response.json()
        # Hugging Face sometimes returns [{'generated_text': ...}]
        corrected_text = data[0]['generated_text'] if isinstance(data, list) else ""
        # Remove prompt repetition if present
        if prompt in corrected_text:
            corrected_text = corrected_text.replace(prompt, "").strip()
        return corrected_text
    else:
        st.error(f"Hugging Face API error: {response.status_code}")
        return None

st.set_page_config(page_title="Handwritten OCR with HuggingFace LLM", page_icon="✍️")
st.title("📝 Handwritten OCR + Hugging Face LLM Correction")

@st.cache_resource
def load_models():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    easy_reader = easyocr.Reader(['en'])
    return processor, model, device, easy_reader

processor, model, device, reader = load_models()

UPLOAD_FOLDER = "uploads"
CORRECTIONS_FOLDER = "corrections"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CORRECTIONS_FOLDER, exist_ok=True)

uploaded_file = st.file_uploader("Upload handwritten image", type=["jpg", "jpeg", "png"])
manual_threshold = st.slider("Binary Threshold (0 disables)", 0, 255, 0)
use_segmentation = st.checkbox("Use line and word segmentation", True)

if uploaded_file:

    base_fname = os.path.splitext(uploaded_file.name)[0]
    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    preprocessed_img = preprocess_image(save_path, manual_threshold if manual_threshold > 0 else None)    
    st.subheader("Preprocessed Image")
    st.image(preprocessed_img, use_column_width=True, channels="GRAY")

    texts = []

    if use_segmentation:
        lines = segment_lines(preprocessed_img)
        st.write(f"Detected {len(lines)} line(s)")

        for idx, line_img in enumerate(lines):
            words = segment_words(line_img)
            st.write(f"Line {idx+1}: Detected {len(words)} word(s)")
            line_text = []
            for i, word_img in enumerate(words):
                pil_word = Image.fromarray(word_img).convert("RGB")

                inputs = processor(pil_word, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(inputs)
                trocr_word = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                easy_results = reader.readtext(word_img)
                easy_word = " ".join(res[1] for res in easy_results)

                chosen = trocr_word if len(trocr_word) > len(easy_word) else easy_word
                line_text.append(chosen)

                # Display word OCR comparison for debugging
                st.write(f"Word {i+1} TrOCR: {trocr_word}")
                st.write(f"Word {i+1} EasyOCR: {easy_word}")

            texts.append(" ".join(line_text))
    else:
        pil_img = Image.fromarray(preprocessed_img).convert("RGB")
        inputs = processor(pil_img, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(inputs)
        trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        easy_results = reader.readtext(preprocessed_img)
        easy_text = " ".join(res[1] for res in easy_results)

        chosen = trocr_text if len(trocr_text) > len(easy_text) else easy_text
        texts.append(chosen)

    raw_text = "\n".join(texts)
    clean_text = filter_ocr_output(raw_text)

    st.subheader("OCR Extracted Text")
    st.text_area("Raw OCR Output", raw_text, height=200)

    st.subheader("Filtered OCR Output")
    st.text_area("Cleaned OCR Output", clean_text, height=200)

    if not HUGGINGFACE_API_KEY:
        st.warning("Hugging Face API key not found in .env file")
    else:
        if st.button("Correct with Hugging Face LLM"):
            corrected_text = hf_llm_correction(clean_text)
            if corrected_text:
                st.success("Text corrected successfully!")
                st.text_area("LLM Corrected Text", corrected_text, height=250)

                save_path = os.path.join(CORRECTIONS_FOLDER, f"{base_fname}_llm_correction.txt")
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(corrected_text)
                st.write(f"Corrected text saved to {save_path}")

    st.subheader("Manual Correction (Optional)")
    user_correction = st.text_area("Please correct OCR or LLM results if needed:", clean_text)

    if st.button("Save Manual Correction"):
        manual_path = os.path.join(CORRECTIONS_FOLDER, f"{base_fname}_manual_correction.txt")
        with open(manual_path, "w", encoding="utf-8") as f:
            f.write(user_correction)
        st.success(f"Manual correction saved at {manual_path}")

    output_file = os.path.join(UPLOAD_FOLDER, f"{base_fname}_final_corrected.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(user_correction or clean_text)

    with open(output_file, "r", encoding="utf-8") as f:
        st.download_button("⬇️ Download Final Corrected Text", f, file_name=f"{base_fname}_final_corrected.txt")
