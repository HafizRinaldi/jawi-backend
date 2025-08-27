import json
import os
import numpy as np
import requests
import faiss
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# KONFIGURASI APLIKASI 
QWEN_API_URL = "https://litellm.bangka.productionready.xyz/v1/chat/completions"
MODEL_NAME = "vllm-qwen3"
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
if not QWEN_API_KEY:
    raise ValueError("API Key (QWEN_API_KEY) tidak ditemukan. Harap set sebagai environment variable.")

# MEMUAT MODEL & DATABASE VEKTOR SAAT SERVER DIMULAI 
print("Memuat model retriever (SentenceTransformer)...")
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Memuat database vektor (FAISS)...")
jawi_index = faiss.read_index('jawi_index.faiss')
print("Memuat dokumen pengetahuan...")
with open('documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)
print("✅ Server siap menerima permintaan!")
print("-" * 30)

# ENDPOINT 1: MODE GURU (FAKTUAL DENGAN RAG) 
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    contextual_topic = data.get('context')
    if not user_query: return jsonify({"error": "Query tidak ditemukan"}), 400

    print(f"\n🚀 Menerima query faktual: '{user_query}' | Konteks: '{contextual_topic}'")
    
    affirmative_words = ['iya', 'ya', 'oke', 'boleh', 'ok', 'lanjut', 'jelaskan', 'iyaa']
    if contextual_topic and user_query.lower().strip() in affirmative_words:
        search_query = contextual_topic
    else:
        search_query = user_query

    query_embedding = retriever_model.encode([search_query])
    distances, indices = jawi_index.search(np.array(query_embedding).astype('float32'), k=3)
    retrieved_context = "\n\n".join([documents[i] for i in indices[0]])
    
    prompt_template = f"""
    Anda adalah JawiAI, asisten AI yang cerdas dan presisi untuk Aksara Jawi.
    IKUTI ATURAN HIERARKI INI:
    1.  ATURAN SAPAAN: Jika pengguna menyapa ('hai', 'hallo'), balas dengan ramah.
    2.  ATURAN KONTEKS AFIRMATIF: Jika pengguna merespons 'iya'/'ya' DAN ada 'Topik Kontekstual', jelaskan 'Topik Kontekstual' itu.
    3.  ATURAN CONTOH KATA: Jika pengguna meminta 'contoh', ekstrak contoh dari 'Konteks'.
    4.  ATURAN FORMAT (PENTING): Saat menjelaskan huruf atau memberi contoh, WAJIB sertakan tulisan Latin dan Jawi. FORMAT: Latin (Jawi). CONTOH: 'banyak' (باڽق).
    5.  ATURAN UMUM: Jawab pertanyaan lain HANYA berdasarkan 'Konteks'.
    6.  ATURAN FALLBACK: Jika tidak ada info, katakan tidak tahu.

    Topik Kontekstual: {contextual_topic}
    Konteks: {retrieved_context} 
    Pertanyaan Pengguna: {user_query}
    Jawaban Anda:
    """

    try:
        headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_template}], "max_tokens": 300, "temperature": 0.5}
        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        ai_response = response_data['choices'][0]['message']['content']
        return jsonify({"response": ai_response.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ENDPOINT 2: MODE KREATIF 
@app.route('/chat-creative', methods=['POST'])
def chat_creative():
    data = request.json
    user_query = data.get('query')
    if not user_query: return jsonify({"error": "Query tidak ditemukan"}), 400

    creative_prompt = f"""
    Anda adalah guru bahasa Jawi yang kreatif. Penuhi permintaan pengguna. Sertakan tulisan Latin dan Jawi jika memungkinkan.
    Permintaan: "{user_query}"
    Jawaban Kreatif Anda:
    """
    try:
        headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": creative_prompt}], "max_tokens": 150, "temperature": 0.8}
        response = requests.post(QWEN_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        ai_response = response_data['choices'][0]['message']['content']
        return jsonify({"response": ai_response.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# MENJALANKAN SERVER 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)