from flask import Flask, render_template, Response, request, redirect, url_for, session ,jsonify
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import pandas as pd
import math
import pyttsx3
import smtplib
from email.message import EmailMessage
import firebase_admin
from firebase_admin import credentials, db
from werkzeug.utils import secure_filename
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
import os
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)  # Adjust number of workers as needed
from datetime import datetime



app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a secure secret key

# Initialize Firebase Admin SDK
cred = credentials.Certificate("fbconfig.json")  # Update with your Firebase credentials file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://projectcg9-942bf-default-rtdb.firebaseio.com/'  # Update with your Firebase Realtime Database URL
})

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

pose_data_df = None
torso_size_multiplier = 2.5
prev_result = ""
training_mode = False


cap = cv2.VideoCapture(0)

# Load pose data if CSV exists
if os.path.exists("pose_data.csv"):
    pose_data_df = pd.read_csv("pose_data.csv")
else:
    pose_data_df = pd.DataFrame()


# Video frame generator
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if email == "admin@gmail.com" and password == "pass":
        session['user'] = "admin"
        return jsonify({"success": True, "redirect": "/admin_home"})

    users_ref = db.reference('Users')
    users_data = users_ref.get()

    if users_data:
        for user_id, user_info in users_data.items():
            if user_info.get("email") == email and user_info.get("password") == password:
                session['user'] = user_id
                session['location'] = user_info.get("location")
                return jsonify({"success": True, "redirect": "/user_home"})

    return jsonify({"success": False, "message": "Invalid credentials or user not registered."})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    contact = data.get('contact')
    location = data.get('location')
    password = data.get('password')

    if not all([name, email, contact, location, password]):
        return jsonify({"success": False, "message": "All fields are required!"})

    users_ref = db.reference('Users')
    users_data = users_ref.get()

    if users_data is not None:
        for user_info in users_data.values():
            if user_info.get('email') == email:
                return jsonify({"success": False, "message": "Email already registered!"})

    new_user_ref = users_ref.push({
        'name': name,
        'email': email,
        'contact': contact,
        'location': location,
        'password': password
    })

    session['user'] = new_user_ref.key
    return jsonify({"success": True, "message": "User registered successfully!", "redirect": "/"})

@app.route('/train_pose')
def train_pose():
    if 'user' not in session:
        return redirect('/login')
    return render_template('train.html')

@app.route('/train_images', methods=['POST', 'GET'])
def train_images():
    if 'user' not in session:
        return redirect('/login')
    return render_template('train_images.html')

@app.route('/upload_folder', methods=['POST'])
def train_from_folder():
    if 'folder' not in request.files:
        return 'No folder images uploaded.', 400

    files = request.files.getlist('folder')
    if len(files) == 0:
        return 'No images received.', 400

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    data = []

    # Prepare column names
    col_names = []
    for name in mp_pose.PoseLandmark.__members__.keys():
        for c in ['X', 'Y', 'Z', 'V']:
            col_names.append(f'{name}_{c}')
    col_names.append('label')

    for file in files:
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        label = filename.split('_')[0]

        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm_list = [landmark for landmark in results.pose_landmarks.landmark]
            landmark_names = list(mp_pose.PoseLandmark.__members__.keys())
            try:
                right_hip = landmark_names.index('RIGHT_HIP')
                left_hip = landmark_names.index('LEFT_HIP')
                right_shoulder = landmark_names.index('RIGHT_SHOULDER')
                left_shoulder = landmark_names.index('LEFT_SHOULDER')
            except ValueError:
                continue

            center_x = (lm_list[right_hip].x + lm_list[left_hip].x) * 0.5
            center_y = (lm_list[right_hip].y + lm_list[left_hip].y) * 0.5
            shoulders_x = (lm_list[right_shoulder].x + lm_list[left_shoulder].x) * 0.5
            shoulders_y = (lm_list[right_shoulder].y + lm_list[left_shoulder].y) * 0.5

            max_distance = max([math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2) for lm in lm_list])
            torso_size = math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2)
            max_distance = max(torso_size * 2.5, max_distance)

            pre_lm = np.array([[(lm.x - center_x) / max_distance,
                                (lm.y - center_y) / max_distance,
                                lm.z / max_distance,
                                lm.visibility] for lm in lm_list]).flatten()

            row = list(pre_lm) + [label]
            data.append(row)

    csv_path = 'pose_data.csv'
    df = pd.DataFrame(data, columns=col_names)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', index=False, header=False)

    pose.close()
    return 'Training data saved successfully from uploaded folder images!'


@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect('/login')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

@app.route('/capture_pose', methods=['POST'])
def capture_pose():
    if 'user' not in session :
        return redirect('/login')

    pose_name = request.form['pose_name']
    success, frame = cap.read()
    if not success:
        return "Failed to capture frame", 500

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        lm_list = [landmark for landmark in results.pose_landmarks.landmark]
        landmark_names = list(mp_pose.PoseLandmark.__members__.keys())
        right_hip = landmark_names.index('RIGHT_HIP')
        left_hip = landmark_names.index('LEFT_HIP')
        right_shoulder = landmark_names.index('RIGHT_SHOULDER')
        left_shoulder = landmark_names.index('LEFT_SHOULDER')

        center_x = (lm_list[right_hip].x + lm_list[left_hip].x) * 0.5
        center_y = (lm_list[right_hip].y + lm_list[left_hip].y) * 0.5

        shoulders_x = (lm_list[right_shoulder].x + lm_list[left_shoulder].x) * 0.5
        shoulders_y = (lm_list[right_shoulder].y + lm_list[left_shoulder].y) * 0.5

        max_distance = max([math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2) for lm in lm_list])
        torso_size = math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2)
        max_distance = max(torso_size * 2.5, max_distance)

        # Generate actual feature names
        col_names = []
        for name in mp_pose.PoseLandmark.__members__.keys():
            for c in ['X', 'Y', 'Z', 'V']:
                col_names.append(f'{name}_{c}')

        pre_lm = np.array([[(lm.x - center_x) / max_distance,
                            (lm.y - center_y) / max_distance,
                            lm.z / max_distance,
                            lm.visibility] for lm in lm_list]).flatten()

        data_df = pd.DataFrame([pre_lm], columns=col_names)
        data_df['label'] = pose_name

        csv_path = 'pose_data.csv'
        if not os.path.exists(csv_path):
            data_df.to_csv(csv_path, index=False)
        else:
            data_df.to_csv(csv_path, mode='a', index=False, header=False)

        return redirect('/train_pose')

    return "No pose detected. Please try again.", 400


def SpeakText(command):
        try:
            engine = pyttsx3.init()
            engine.say(command)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS ERROR] {e}")


def send_email(activity_label, location, frame=None):
    try:
        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = f"activity_alerts/activity_{timestamp}.jpg"

        # Save the frame if available
        if frame is not None:
            cv2.imwrite(image_filename, frame)

        # Compose email
        msg = EmailMessage()
        msg.set_content(f"⚠️ Abnormal Activity Detected: {activity_label}\n📍 Location: {location}\n🕒 Time: {timestamp}")
        msg['Subject'] = f'ALERT: {activity_label} Detected'
        msg['From'] = os.getenv("EMAIL_SENDER")
        msg['To'] = 'three.emergingtech@gmail.com'

        # Attach image if available
        if frame is not None:
            with open(image_filename, 'rb') as img_file:
                img_data = img_file.read()
                msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_filename))

        # Send email
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASS"))
        server.send_message(msg)
        server.quit()
        print("[INFO] Email sent with attachment.")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")



def detect_pose_live_custom(location):
    global prev_result, training_mode

    if pose_data_df is None:
        raise RuntimeError("Training data not loaded.")

    r = len(pose_data_df)

    while True:
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm_list = results.pose_landmarks.landmark

            # Center and normalization
            center_x = (lm_list[24].x + lm_list[23].x) * 0.5
            center_y = (lm_list[24].y + lm_list[23].y) * 0.5
            shoulders_x = (lm_list[12].x + lm_list[11].x) * 0.5
            shoulders_y = (lm_list[12].y + lm_list[11].y) * 0.5

            torso_size = math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2)
            max_distance = max(
                torso_size * torso_size_multiplier,
                max(math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2) for lm in lm_list)
            )

            pre_lm = normalize_landmarks(lm_list, center_x, center_y, max_distance)

            # Manual KNN
            min_dist = float('inf')
            detected_label = "unknown"
            for i in range(r):
                dist = sum(abs(pre_lm[j] - pose_data_df.iloc[i, j]) for j in range(132))
                if dist < min_dist:
                    min_dist = dist
                    detected_label = pose_data_df.iloc[i, 132]

            if min_dist > 8:
                detected_label = "unknown"

            # Show prediction on screen
            cv2.putText(frame, f'Detected: {detected_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.putText(frame, f'Distance: {min_dist:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 0), 2)

            # Alert logic
            if detected_label != prev_result:
               if detected_label != "unknown":
                    print(f"[INFO] Activity changed: {prev_result} → {detected_label}")
                    # You can optionally trigger: 
                    #SpeakText(detected_label)
                    #send_email(detected_label, location, frame)
                    executor.submit(send_email, detected_label, location, frame)
                    executor.submit(SpeakText, detected_label)


                    #send_email(detected_label)
                    prev_result = detected_label

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/live_detect')
def live_detect():
    if 'user' not in session:
        return redirect('/login')
    location = session.get('location','unknown')
    return Response(detect_pose_live_custom(location), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/user_home')
def admin_home():
    if 'user' not in session:
        return redirect('/login')
    return render_template('user_home.html')

@app.route('/admin_home')
def user_home():
    if 'user' not in session:
        return redirect('/login')
    return render_template('admin_home.html')


# Load the pose data once
if os.path.exists("pose_data.csv"):
    pose_data_df = pd.read_csv("pose_data.csv")
    print(f"[INFO] Loaded training data with {len(pose_data_df)} samples.")
else:
    print("[WARNING] Training data 'pose_data.csv' not found.")

def normalize_landmarks(lm_list, center_x, center_y, max_distance):
    return np.array([[(lm.x - center_x) / max_distance,
                      (lm.y - center_y) / max_distance,
                      lm.z / max_distance,
                      lm.visibility] for lm in lm_list]).flatten()


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        video = request.files['video']
        if video:
            filepath = os.path.join('static', 'uploaded_video.mp4')
            video.save(filepath)
            return redirect('/detect_uploaded_video')  # NEW route for live-style detection
    return render_template('upload_video.html')

@app.route('/detect_uploaded_video')
def detect_uploaded_video():
    if 'location' in session:
        location = session['location']
        return Response(gen_from_uploaded_video(location), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Location not found in session.", 400


def gen_from_uploaded_video(location):
    cap = cv2.VideoCapture("static/uploaded_video.mp4")
    global prev_result, training_mode

    if pose_data_df is None:
        raise RuntimeError("Training data not loaded.")

    r = len(pose_data_df)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm_list = results.pose_landmarks.landmark

            # Center and normalization
            center_x = (lm_list[24].x + lm_list[23].x) * 0.5
            center_y = (lm_list[24].y + lm_list[23].y) * 0.5
            shoulders_x = (lm_list[12].x + lm_list[11].x) * 0.5
            shoulders_y = (lm_list[12].y + lm_list[11].y) * 0.5

            torso_size = math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2)
            max_distance = max(
                torso_size * torso_size_multiplier,
                max(math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2) for lm in lm_list)
            )

            pre_lm = normalize_landmarks(lm_list, center_x, center_y, max_distance)

            # Manual KNN
            min_dist = float('inf')
            detected_label = "unknown"
            for i in range(r):
                dist = sum(abs(pre_lm[j] - pose_data_df.iloc[i, j]) for j in range(132))
                if dist < min_dist:
                    min_dist = dist
                    detected_label = pose_data_df.iloc[i, 132]

            if min_dist > 8:
                detected_label = "unknown"

            # Show prediction on screen
            cv2.putText(frame, f'Detected: {detected_label}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3)
            cv2.putText(frame, f'Distance: {min_dist:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 255, 0), 3)

            # Alert logic
            if detected_label != prev_result:
               if detected_label != "unknown":
                    print(f"[INFO] Activity changed: {prev_result} → {detected_label}")
                    # You can optionally trigger: 
                    executor.submit(send_email, detected_label, location, frame)
                    executor.submit(SpeakText, detected_label)
                    prev_result = detected_label

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Start app
if __name__ == '__main__':
    app.run(debug=True)
