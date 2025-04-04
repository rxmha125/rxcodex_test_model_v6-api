from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time  # Added for timing

app = Flask(__name__)

# Hugging Face model URL and token
MODEL_URL = "rxmha125/rxcodex_test_model_v6-test"
HF_TOKEN = "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"  # Replace with your token
API_KEY = "your-secret-api-key"  # Replace with your API key

# Load model and tokenizer with token
tokenizer = AutoTokenizer.from_pretrained(MODEL_URL, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_URL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate response with timing
def generate_response(prompt):
    start_time = time.time()  # Start timer
    inputs = tokenizer(f"User: {prompt}\nBot:", return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        no_repeat_ngram_size=2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response.replace(f"User: {prompt}\nBot:", "").strip()
    end_time = time.time()  # End timer
    generation_time = end_time - start_time  # Time in seconds
    return response_text, generation_time

# API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    provided_key = request.headers.get("X-API-Key")
    if provided_key != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
    
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing prompt"}), 400
    
    prompt = data["prompt"]
    response, gen_time = generate_response(prompt)
    return jsonify({
        "response": response,
        "generation_time_seconds": round(gen_time, 2)  # Rounded to 2 decimals
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)