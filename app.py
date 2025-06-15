import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-te'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    gen = model.generate(**batch)
    return tokenizer.decode(gen[0], skip_special_tokens=True)

# Streamlit UI
st.title("English to Telugu Translator")
input_text = st.text_area("Enter English text")

if st.button("Translate"):
    if input_text.strip():
        output_text = translate(input_text)
        st.success(f"Telugu Translation:\n{output_text}")
    else:
        st.warning("Please enter some text.")
