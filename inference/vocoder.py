import torch
import soundfile as sf # Sesi .wav olarak kaydetmek için

class VocoderInference:
    import torch

class VocoderInference:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"Loading Pre-trained Universal HiFi-GAN on {self.device}...")
        
        # HATA BURADAYDI: map_location ekleyerek modelin CPU'ya inmesini sağlıyoruz
        self.vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan', map_location=torch.device(device)).to(self.device)
        
        self.vocoder.eval()

    def generate_audio(self, mel_spectrogram, output_path="cloned_voice.wav"):
        # mel_spectrogram shape: (1, 80, Sequence_Length)
        mel_spectrogram = mel_spectrogram.to(self.device)
        
        # torch.no_grad() -> Gradient hesaplamasını kapatır. 
        # PC'ni yormaz, 16GB VRAM'in sadece çok küçük bir kısmını kullanır ve çok hızlı çalışır.
        with torch.no_grad():
            # Matrisi veriyoruz, sesi alıyoruz.
            audio_waveform = self.vocoder(mel_spectrogram)
            
        # Sesi CPU'ya çekip 1D array'e düzleştiriyoruz (hoparlörün çalabilmesi için)
        audio_numpy = audio_waveform.squeeze().cpu().numpy()
        
        # 22.05kHz sample rate ile diske kaydediyoruz
        sf.write(output_path, audio_numpy, 22050)
        print(f"Audio saved to {output_path} successfully!")
        
        return audio_numpy
