
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from transformers import pipeline as classification_pipeline
import torch
import os
import re
from pydantic import BaseModel

# Define paths to save the model and tokenizer
MODEL_PATH = "./saved_model"
TOKENIZER_PATH = "./saved_tokenizer"

VALID_TOPICS = {"math", "coding", "physics", "chemistry", "theory"}

# Initialize the model and tokenizer
def load_model_and_tokenizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Loading model and tokenizer from saved files...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    else:
        print("Downloading and saving model and tokenizer...")
        model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map='auto'
        )
        # Save the model and tokenizer
        tokenizer.save_pretrained(TOKENIZER_PATH)
        model.save_pretrained(MODEL_PATH)

    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=512)

classifier = classification_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Create the FastAPI app
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate(request: PromptRequest):
    # Classify the prompt topic
    classification_result = classifier(request.prompt, list(VALID_TOPICS))
    topic = classification_result["labels"][0]
    confidence = classification_result["scores"][0]

    # Validate the topic
    if topic not in VALID_TOPICS or confidence < 0.5:
        return {
            "error": "Invalid or low-confidence topic detected.",
            "detected_topic": topic,
            "confidence": confidence
        }

    # Generate the response
    response = text_generator(request.prompt)
    generated_text = response[0]['generated_text']

    # Extract code snippets within triple backticks
    code_snippets = re.findall(r"'''(.*?)'''", generated_text, re.DOTALL)

    return {
        "detected_topic": topic,
        "confidence": confidence,
        "generated_text": generated_text,
        "code_snippets": code_snippets
    } 
    