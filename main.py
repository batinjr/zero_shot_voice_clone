import os
import torch
import sounddevice as sd


import numpy as np

def trim_silence(audio_data, threshold=0.02, margin_ms=100, sample_rate=24000):
    """
    Advanced VAD Trimmer. 
    It cuts empty space but keeps a small margin to save the end of the words.
    """
    # 1. Find all sound above the threshold (0.02 is better for breaths)
    active_indices = np.where(np.abs(audio_data) > threshold)[0]
    
    if len(active_indices) > 0:
        # 2. Calculate the margin in array length (samples)
        # 100ms margin at 24000Hz = 2400 array items
        margin_samples = int((margin_ms / 1000.0) * sample_rate)
        
        # 3. Add the margin safely (do not go below 0 or above array length)
        start = max(0, active_indices[0] - margin_samples)
        end = min(len(audio_data), active_indices[-1] + margin_samples)
        
        return audio_data[start:end]
    else:
        return audio_data
# --- 2. AUDIO BACKEND FIX (Bu kalsın, ses dosyası okumak için lazım) ---
import torchaudio
import soundfile as sf

def _custom_torchaudio_load(filepath, **kwargs):
    wav, sr = sf.read(filepath, dtype='float32')
    if wav.ndim == 1:
        tensor = torch.from_numpy(wav).unsqueeze(0)
    else: 
        tensor = torch.from_numpy(wav).T
    return tensor, sr

def _custom_torchaudio_info(filepath, **kwargs):
    info = sf.info(filepath)
    class MetaData:
        sample_rate = info.samplerate
    return MetaData()

torchaudio.load = _custom_torchaudio_load
torchaudio.info = _custom_torchaudio_info
# ----------------------------

from TTS.api import TTS
import requests
import re
import threading
import queue
import json

os.environ["COQUI_TOS_AGREED"] = "1"


# 1. THE QUEUE SYSTEM
# This is the "box" where we put ready audio files
audio_queue = queue.Queue()

def audio_worker():
    """Background Thread: RAM'den gelen saf ses matrislerini kesintisiz çalar."""
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break
            
        # Sesi çal, ama bitmesini milisaniyelik hassasiyetle yönetmek için sd.play sonrası doğrudan kanal takibi yapıyoruz
        sd.play(audio_data, samplerate=24000)
        sd.wait() 
        audio_queue.task_done()
def stream_llm_sentences(user_text):
    print("Bot is thinking... (Streaming Mode ON)")
    url = "http://localhost:11434/api/generate"
    
    strict_prompt = (
        "Senin tek dilin Türkçe. Sadece TÜRKÇE konuşacaksın. "
        "Kısa cümleler kur. Asla sayı, rakam veya liste kullanma. "
        f"Kullanıcı: {user_text}"
    )
    
    payload = {
        "model": "llama3",
        "prompt": strict_prompt,
        "stream": True,
        "options": {"temperature": 0.3}
    }
    
    current_sentence = ""
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    word = data.get("response", "")
                    current_sentence += word
                    
                    # We only yield when we see a real sentence end (. ! ?)
                    if any(punc in word for punc in ['.', '!', '?']):
                        clean = re.sub(r'[0-9]+', '', current_sentence)
                        # THE COMMA FIX: We removed the comma (,) from the allowed characters
                        clean = re.sub(r'[^\w\s.!?çğıöşüÇĞİÖŞÜ]', '', clean)
                        clean = clean.strip()
                        
                        if clean:
                            yield clean # Send ONE clean sentence directly
                            
                        current_sentence = "" 
                        
    except Exception as e:
        print(f"Ollama stream error: {e}")

# 3. THE MAIN PIPELINE
def main():
    print("System is starting...")
    device = "cpu" 
    print("Loading XTTS Engine...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    voice_file = "senin_sesin.wav" 
    
    player_thread = threading.Thread(target=audio_worker, daemon=True)
    player_thread.start()

    print("\nSystem READY! (Type 'exit' to quit)")
    print("-" * 30)

    while True:
        user_input = input("\nSen: ")
        if user_input.lower() in ['exit', 'quit', 'cikis']:
            break

        for sentence in stream_llm_sentences(user_input):
            print(f"[XTTS Engine] Synthesizing: {sentence}")
            
            wav_array = tts.tts(
                text=sentence, 
                speaker_wav=voice_file, 
                language="tr", 
                speed=1.15
            )
            
            # 1. Listeyi numpy matrisine çevir
            audio_data = np.array(wav_array)
            
            # 2. THE CORE FIX: Başındaki ve sonundaki nefes/boşluk seslerini makasla
            audio_data = trim_silence(audio_data, threshold=0.015)
            
            # 3. Temizlenmiş saf sesi çalınması için sıraya at
            audio_queue.put(audio_data)

if __name__ == "__main__":
    main()