import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

print("Memulai proses ingestion data...")

# Membaca basis pengetahuan utama dari file JSON
with open('jawi_knowledge.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

# Mempersiapkan list kosong untuk menampung teks yang akan diproses
documents = []

# Memproses setiap item di dalam basis pengetahuan
for item in knowledge_base:
    if item.get('type') == 'topik_umum':
        # Format teks untuk topik umum
        text = f"Topik: {item['topik']}. Penjelasan: {item['konten']}"
        documents.append(text)
    elif item.get('type') == 'huruf':
        # Format teks untuk data huruf, menyertakan semua detail
        # termasuk karakter Jawi dan contoh ejaan Jawi.
        text = (f"Nama huruf: {item['nama']}. "
                f"Bentuk karakter Jawi: {item.get('karakter', '')}. "
                f"Info: {item['info']} "
                f"Contoh kata dalam latin adalah '{item.get('contoh_latin', '')}' "
                f"dan dalam Jawi adalah '{item.get('contoh_jawi', '')}'.")
        documents.append(text)

print(f"Total {len(documents)} dokumen akan diproses.")

# Memuat model untuk mengubah teks menjadi vektor (embedding)
print("Memuat model SentenceTransformer 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Proses encoding: mengubah semua teks menjadi representasi angka (vektor)
print("Membuat embeddings untuk semua dokumen...")
embeddings = model.encode(documents, convert_to_tensor=False)

# Membangun dan menyimpan database vektor menggunakan FAISS
print("Membangun dan menyimpan index ke file 'jawi_index.faiss'...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))
faiss.write_index(index, 'jawi_index.faiss')

# Menyimpan dokumen teks yang sudah diformat untuk referensi
print("Menyimpan dokumen teks ke file 'documents.json'...")
with open('documents.json', 'w', encoding='utf-8') as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)

print("✅ Proses ingestion selesai! Database vektor dan dokumen telah diperbarui.")