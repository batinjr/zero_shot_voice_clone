import requests

def get_llm_response(user_text):
    # This is the local Ollama server address
    url = "http://localhost:11434/api/generate"
    
    # We send the model name and user text
    payload = {
        "model": "llama3", # We will pull this model later
        "prompt": f"You are a helpful voice assistant. Keep it short. User: {user_text}",
        "stream": False # Wait for the full answer
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Check for HTTP errors
        
        # Parse the JSON and get the text
        data = response.json()
        llm_text = data["response"]
        return llm_text
        
    except Exception as e:
        print(f"Ollama connection error: {e}")
        return None

# Test the LLM Engine
reply = get_llm_response("Bana kendini tanıtır mısın?")
print(f"LLM Reply: {reply}")
