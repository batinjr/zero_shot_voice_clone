import os
import torch

# --- PYTORCH 2.6 GÜVENLİK KİLİDİNİ KIRMA YAMASI ---
_original_load = torch.load 

def _custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _custom_load
# --------------------------------------------------

# --- SES MOTORU YAMASI (THE ULTIMATE AUDIO PATCH) ---
import torchaudio
import soundfile as sf
import torch

# PyTorch'un bozuk okuyucusunu kendi fonksiyonumuzla eziyoruz
def _custom_torchaudio_load(filepath, **kwargs):
    wav, sr = sf.read(filepath, dtype='float32')
    # Ses mono ise (tek kanal), modelin istediği 2D [1, Time] matrisine çeviriyoruz
    if wav.ndim == 1:
        tensor = torch.from_numpy(wav).unsqueeze(0)
    else: # Stereo ise kanalları ayır
        tensor = torch.from_numpy(wav).T
    return tensor, sr

def _custom_torchaudio_info(filepath, **kwargs):
    # Model bazen dosyanın örnekleme hızını (sample rate) sorar, sahte bir obje dönüyoruz
    info = sf.info(filepath)
    class MetaData:
        sample_rate = info.samplerate
    return MetaData()

# RAM üzerinde fonksiyonları değiştiriyoruz
torchaudio.load = _custom_torchaudio_load
torchaudio.info = _custom_torchaudio_info
# ----------------------------------------------------
# -------------------------

from TTS.api import TTS
import requests
import winsound

os.environ["COQUI_TOS_AGREED"] = "1"


import requests
import re # We need this for the new text filter

def get_llm_response(user_text):
    print("Bot is thinking... (Ollama Llama-3)")
    url = "http://localhost:11434/api/generate"
    
    strict_prompt = (
        "Senin tek dilin Türkçe. Sadece TÜRKÇE konuşacaksın. "
        "Sadece 2 veya 3 cümle ile cevap ver. Uzun paragraf yazma. "
        "Asla sayı, rakam veya liste kullanma."
        f"Kullanıcı: {user_text}"
    )
    
    payload = {
        "model": "llama3",
        "prompt": strict_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3 
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        raw_text = response.json()["response"]
        
        # 1. Clean basic brackets and new lines
        clean_text = raw_text.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
        clean_text = clean_text.replace('\n', ' ')
        
        # 2. THE NEW FIX: Remove all numbers and weird symbols. 
        # Only keep Turkish letters, spaces, and basic punctuation.
        clean_text = re.sub(r'[0-9]+', '', clean_text) 
        clean_text = re.sub(r'[^\w\s.,!?çğıöşüÇĞİÖŞÜ]', '', clean_text)
        
        clean_text = clean_text.strip()
        
        # 3. Fix the punctuation
        if not clean_text.endswith(('.', '!', '?')):
            clean_text += '.'
            
        return clean_text
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

def main():
    print("System is starting...")
    # Güvenlik için şimdilik CPU'da başlatıyoruz, model çok ağır değil
    device = "cpu" 
    
    # 1. Load the Pre-trained Acoustic Model & Vocoder (XTTS)
    # Bu model ilk çalışmada yaklaşık 2 GB ağırlık indirecek.
    print("Loading XTTS Zero-Shot Voice Engine...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    voice_file = "senin_sesin.wav" # Buraya kendi dosyanın adını yaz
    output_file = "bot_answer.wav"

    print("System READY! (Type 'exit' to quit)")
    print("-" * 30)

    # 2. The Real-time Chat Loop
    while True:
        user_input = input("\nSen: ")
        if user_input.lower() in ['exit', 'quit', 'cikis']:
            print("Shutting down...")
            break

        # A. Metni LLM'den al
        llm_text = get_llm_response(user_input)
        if not llm_text:
            continue
        print(f"Bot Text: {llm_text}")

        # B. Ses Klonlama (DNA Extraction + Acoustic + Vocoder in one step!)
        print("Synthesizing voice with your DNA...")
        # XTTS modeli arka planda tam olarak bizim yazdığımız matrisleri kullanıyor
        tts.tts_to_file(text=llm_text, speaker_wav=voice_file, language="tr", file_path=output_file,speed=1.15)
        
        # C. Sesi otomatik çal (Artık dosyaya tıklamana gerek yok)
        print("Playing audio...")
        winsound.PlaySound(output_file, winsound.SND_FILENAME)

if __name__ == "__main__":
    main()