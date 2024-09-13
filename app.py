from flask import Flask, render_template, request, Response, jsonify
import subprocess
import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from text_processing import extract_keywords_from_text
from model_prediction import predict_violence
import time
import subprocess
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Global variables to store results
generated_caption = ""
violence_result = ""

def generate_frames():
    global generated_caption, violence_result
    cap = cv2.VideoCapture(0)  # Use device index for webcam
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_path = f"static/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)

            try:
                raw_image = Image.open(frame_path).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt")
                out = model.generate(**inputs)
                generated_caption = processor.decode(out[0], skip_special_tokens=True)
                raw_image.close()  # Close the image file
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

            keywords_file = f"static/keywords_{frame_count}.txt"
            extract_keywords_from_text(generated_caption, keywords_file)
            violence_result = predict_violence(keywords_file)

            frame_count += 1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            time.sleep(0.1)  # Add a short delay to avoid overloading the client
    finally:
        cap.release()  # Ensure the capture is released when done

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video uploaded', 400
    
    video_file = request.files['video']
    video_file.save('uploaded_video.mp4')
    
    # Appeler le script main.py avec le chemin de la vidéo
    def run_analysis_and_emit_progress():
        process = subprocess.Popen(["python", "main.py", "uploaded_video.mp4"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while process.poll() is None:
            # Simulate progress
            for i in range(0, 101, 10):
                time.sleep(1)
                socketio.emit('progress', {'percent': i})
        
        stdout, stderr = process.communicate()
        socketio.emit('progress', {'percent': 100})  # Ensure progress reaches 100%
        return stdout.strip()

    result = run_analysis_and_emit_progress()
    print(f"Result from main.py: {result}")  # Debug line
    
    try:
        decoded_result = result.decode('utf-8')
    except UnicodeDecodeError:
        decoded_result = result.decode('utf-8', errors='replace')  # Replace undecodable bytes

    return jsonify({"result": decoded_result})

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results():
    global generated_caption, violence_result
    return {
        "result": violence_result
    }
@app.route('/singin')
def singin():
    return render_template('singin.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)  # Use socketio.run to enable SocketIO