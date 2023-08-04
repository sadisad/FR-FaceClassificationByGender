# Face Recognition

## Latar Belakang 
Face Recognition Gender Classification penting karena memiliki beberapa aplikasi dan implikasi yang relevan:

1. Identifikasi Kriminal: Gender classification dalam face recognition dapat membantu lembaga penegak hukum dalam mengidentifikasi pelaku kejahatan berdasarkan rekaman wajah dari kamera pengawas. Kemampuan ini membantu dalam pelacakan dan penangkapan pelaku kriminal, membantu meningkatkan keamanan masyarakat, dan memberikan kontribusi dalam penegakan hukum.

2. Analisis Demografi: Gender classification dalam face recognition juga dapat digunakan untuk analisis demografi dalam berbagai konteks. Misalnya, dalam analisis pasar atau iklan, informasi tentang perbandingan gender dalam pengguna atau pelanggan dapat membantu perusahaan dalam merancang strategi pemasaran yang lebih tepat sasaran.

3. Identifikasi Korban Hilang: Face recognition gender classification dapat membantu dalam identifikasi korban hilang atau orang yang hilang. Dengan teknologi ini, orang-orang yang hilang dapat diidentifikasi lebih cepat, dan upaya pencarian dan penyelamatan dapat ditingkatkan.

4. Penelitian Psikologi: Teknologi gender classification dalam face recognition juga dapat digunakan dalam penelitian psikologi dan perilaku manusia. Data tentang perbedaan reaksi dan perilaku antara laki-laki dan perempuan dalam berbagai situasi dapat digunakan untuk memahami pola-pola psikologis dan perilaku manusia.

5. Analisis dan Pemetaan Kehadiran: Dalam konteks bisnis dan pendidikan, teknologi gender classification dapat digunakan untuk analisis dan pemetaan kehadiran. Misalnya, dalam pendidikan, teknologi ini dapat membantu mengidentifikasi perbedaan antara kehadiran siswa laki-laki dan perempuan di sekolah.

![main-qimg-d49f541d29c140ee5f604514c26616d5-lq](https://github.com/sadisad/FR-FaceClassificationByGender/assets/61278337/add68dc9-57ce-49c5-b3b2-4108998e9455)

## Gender Classification menggunakan VGG, ResNet and GoogleNet
Dalam repository ini saya dan kelompok yakni Elon Musk dari Indonesia AI Computer Vision Batch 1 melakukan implementasi model beberapa model deep learning untuk klasifikasi jenis kelamin (gender) dengan datasets CelebA. Model Deep Learning yang kami coba implementasikan yakni:

- [VGG16](https://github.com/sadisad/FR-FaceClassificationByGender/tree/main/VGG16). (Take Over By Irsyad Dzulfikar)
- [VGG19](https://github.com/sadisad/FR-FaceClassificationByGender/tree/main/VGG-19). (Take Over By Asrul Said)
- [ResNet-18](https://github.com/sadisad/FR-FaceClassificationByGender/tree/main/ResNet_18) (Take Over By Daniel Riandy)
- [Google Net](https://github.com/sadisad/FR-FaceClassificationByGender/tree/main/Google-Net). (Take Over By Bayu Dwi Prasetya)

## Kesimpulan
- Learning rate : Semakin kecil learning rate, loss functionya semakin kecil, namun untuk proses waktu training semakin lama
- Batch_size : Semakin besar, semakin cepat proses training, namun loss function semakin besar
- epoch : Semakin banyak semakin bisa untuk belajar atau training, namun ketika tidak ada perubahan pada akurasi tiap epoch nya, maka besar epoch tidak terlalu berimbas pada model
- split training test 80:20; 70:30; dan 60:40; : Tidak memberikan pengaruh yang signifikan pada akurasi model
- Best parameter : parameter terbaik yang mendapatkan akurasi paling tinggi adalah learning rate 0.003, split data 80:20, batch size: 64 dan epoch 10 mendapatkan akurasi paling tinggi sebesar **0.9465**.
