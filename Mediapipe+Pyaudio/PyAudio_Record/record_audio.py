import pyaudio
import wave
import datetime
import os

# Konfigurasi
FORMAT = pyaudio.paInt16  # Format audio
CHANNELS = 1              # Mono
RATE = 44100              # Sample rate
CHUNK = 1024              # Ukuran buffer
RECORD_SECONDS = 2        # Lama perekaman (detik)

# Buat nama file berdasarkan waktu
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"suara_{timestamp}.wav"

# Inisialisasi PyAudio
audio = pyaudio.PyAudio()

# Mulai merekam
print("[INFO] Merekam suara...")
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("[INFO] Selesai merekam")

# Stop dan tutup stream
stream.stop_stream()
stream.close()
audio.terminate()

# Simpan sebagai .wav
with wave.open(filename, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"[INFO] File audio disimpan sebagai: {filename}")
