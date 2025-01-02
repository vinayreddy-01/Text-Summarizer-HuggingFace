import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Ensure the model is on the correct device
device = "cuda" if model.device.type == "cuda" else "cpu"
model = model.to(device)


# Clean text function
def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text


# Summarization function
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_text(dialogue)
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate summary
    outputs = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


# Streamlit interface
st.set_page_config(page_title="Text Summarization System", page_icon="📝", layout="wide")
st.title("Text Summarization System ")
st.markdown(""" 
    <style>
        .title {
            font-size: 2.5rem;
            font-weight: bold;
            color: red; /* This changes the title to red */
        }
        .subheader {
            font-size: 1.5rem;
            color: #4f5d73;
        }
        .text_area {
            background-color: #f6f6f6;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .submit_button {
            background-color: #7c8fe3;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Description
st.markdown("<h2 class='title'>Fine-tuned Hugging Face T5 transformer</h2>", unsafe_allow_html=True)
st.markdown("##### Enter the dialogue below :")

# Input section with styling
dialogue_input = st.text_area(
    "Enter Dialogue", 
    height=200, 
    placeholder="Paste your dialogue here...",
    key="dialogue_input",
    max_chars=2000,
    label_visibility="hidden",
    help="Please input the text you want to summarize.",
)

# When the user clicks on the 'Summarize' button with styling
col1, col2 = st.columns([2, 1])

with col1:
    summarize_button = st.button(
        "Summarize", 
        key="summarize_button",
        help="Click to generate the summary",
        use_container_width=True
    )

with col2:
    st.empty()  # Keeps the column responsive without extra content

if summarize_button:
    if dialogue_input.strip():
        with st.spinner('Generating Summary...'):
            summary = summarize_dialogue(dialogue_input)
            st.success("Summary Generated!")
            st.subheader("Summary:", anchor="summary")
            st.markdown(f"### **{summary}**")
    else:
        st.error("Please enter a dialogue to summarize!")

# Footer with additional information
st.markdown(""" 
    <style>
        footer {
            font-size: 0.85rem;
            color: #888888;
            font-style: italic;
        }
    </style>
    <footer>
        Ta-Da!! - Your Assistant for NLP tasks
    </footer>
""", unsafe_allow_html=True)
