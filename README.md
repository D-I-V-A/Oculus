# OCULUS - Object Detection Library


**Oculus** adalah library Python untuk melakukan **deteksi objek** menggunakan model berbasis **ONNX** dan **ONNX Runtime**.
Tujuan utama dari pengembangan Oculus adalah menyediakan solusi **ringan**, **cepat**, dan **portable** yang dapat berjalan secara optimal bahkan pada **perangkat CPU-only**, tanpa ketergantungan langsung pada framework deep learning seperti PyTorch atau TensorFlow.

## Fitur Utama
- 🔹 Mendukung berbagai model deteksi objek dalam format **ONNX**
- 🔹 Inferensi cepat menggunakan **ONNX Runtime**
- 🔹 Tidak memerlukan GPU — **CPU-only friendly**
- 🔹 Implementasi preprocessing & postprocessing otomatis (resize, NMS, scaling)
- 🔹 Hasil deteksi dapat divisualisasikan langsung dengan OpenCV

## Instalisasi and Requirment Technology

1️⃣ Check Requirment

```sh
python --version  # for window
python3 --version # for linux and mac
```
minimal python versi yang di install adaalah
```sh
> python3.8.0
```
2️⃣ Installization Package
```bash
pip install git+https://github.com/username/Oculus.git
```


🤝 Kontribusi

Kontribusi sangat diterima!
Jika ingin menambahkan fitur baru, memperbaiki bug, atau meningkatkan dokumentasi:

1. Fork repository ini

2. Buat branch baru (feature/nama-fitur)

3. Kirim Pull Request

🔮 Next Features (Roadmap)

- [ ] library ini kompitebel dengan linux
- [ ] libray ini kompitabel dengan google colab
