# program_vision

## Cara Pakai
1. Install requirements.txt
```py
pip install -r requirements.txt
```

2. Install Torch + Cuda Support
```py
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

3. Edit main.py
- Line 30: Model yang dipakai (`depan1.pt` untuk kamera depan, `omni1.pt` untuk kamera omni)
- Line 115: Inisialisasi serial, edit COM berapa
- Line 110 & 111: Seberapa cepat tangkapan metode cadangan (deteksi warna)
- Line 259: Uncomment untuk serial

Kedua line ini musti di comment salah satu sesuai kamera yang dipakai
- Line 245: titik_tengah kamera OMNI
- Line 246: titik_tengah kamera depan

