import streamlit as st
import torch
from transformers import pipeline

def generate_text(input_text):
    generate_text = pipeline(model="databricks/dolly-v2-7b", 
                             torch_dtype=torch.bfloat16, 
                             trust_remote_code=True, 
                             device_map="auto"
                             )
    return generate_text(input_text)

def main():
    st.title("Text Generation App")

    input_text = st.text_input("Enter your text here:")
    if st.button("Generate Text"):
        output_text = generate_text(input_text)
        st.write(output_text)

if __name__ == "__main__":
    main()
