from flask import Flask, render_template, Response, send_file, jsonify
import cv2
import face_recognition
import numpy as np
import os
import time
import subprocess

app = Flask(__name__)

# Konfigurasi path untuk dataset dan dokumen
DATASET_PATH = "dataset"
DOCUMENT_PATH = "document"  # Folder tempat dokumen disimpan
DOCUMENT_NAME = "your_document.pdf"  # Nama dokumen yang akan ditampilkan

# Pastikan folder dataset dan dokumen ada
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(DOCUMENT_PATH, exist_ok=True)

# Muat dataset wajah
def load_known_faces_from_videos(dataset_path, sample_interval=30):
    """Muati encoding wajah dari semua video di folder dataset_path."""
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(dataset_path, filename)
            print(f"Memproses video: {filename}")
            encodings = extract_encodings_from_video(video_path, sample_interval)

            known_face_encodings.extend(encodings)
            name = os.path.splitext(filename)[0]
            known_face_names.extend([name] * len(encodings))  # Satu nama untuk setiap encoding

    return known_face_encodings, known_face_names

def extract_encodings_from_video(video_path, sample_interval=30):
    """Ekstrak encoding wajah dari video."""
    video_capture = cv2.VideoCapture(video_path)
    encodings = []

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Ambil sampel setiap beberapa frame (berdasarkan sample_interval)
        if frame_count % sample_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Simpan semua encoding dari frame ini
            encodings.extend(face_encodings)

        frame_count += 1

    video_capture.release()
    return encodings

# Muat dataset di awal
print("Memuat dataset dari video...")
known_face_encodings, known_face_names = load_known_faces_from_videos(DATASET_PATH)

# Simpan status verifikasi
verified_faces = set()  # Menyimpan nama wajah yang telah diverifikasi
cooldown_period = 5  # Interval waktu untuk reset
last_open_time = 0

# Rute utama untuk halaman web
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint untuk streaming kamera dengan deteksi wajah
@app.route("/video_feed")
def video_feed():
    def generate_frames():
        video_capture = cv2.VideoCapture(0)
        global last_open_time

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Konversi frame ke RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Deteksi wajah dan encoding di frame saat ini
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Bandingkan wajah yang terdeteksi dengan dataset
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    current_time = time.time()

                    if name not in verified_faces or (current_time - last_open_time > cooldown_period):
                        verified_faces.add(name)
                        last_open_time = current_time
                        print(f"Wajah dikenali: {name}")
                else:
                    name = "Tidak dikenali"

                # Tampilkan hasil di layar
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode frame sebagai JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Kirim frame sebagai stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        video_capture.release()

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Rute untuk memeriksa verifikasi
@app.route("/check_verification")
def check_verification():
    if len(verified_faces) > 0:
        return jsonify({"status": "verified", "document_url": "/get_document"})
    return jsonify({"status": "unverified"})

# Rute untuk menyajikan dokumen
@app.route("/get_document")
def get_document():
    document_file = os.path.join(DOCUMENT_PATH, DOCUMENT_NAME)
    if os.path.exists(document_file):
        return send_file(document_file, as_attachment=False)
    return "Dokumen tidak ditemukan.", 404

if __name__ == "__main__":
    app.run(debug=True)
