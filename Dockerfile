# Gunakan base image Python yang ringan
FROM python:3.9-slim

# Tetapkan direktori kerja di dalam kontainer
WORKDIR /app

# Salin file requirements terlebih dahulu untuk caching yang lebih baik
COPY requirements.txt .

# Instal semua library yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file dari folder backend Anda ke dalam kontainer
# Ini termasuk app.py, jawi_knowledge.json, jawi_index.faiss, dll.
COPY . .

# Beritahu Docker port mana yang akan diekspos oleh aplikasi
EXPOSE 8080

# Perintah untuk menjalankan aplikasi menggunakan Gunicorn
# Back4App biasanya menggunakan port 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
