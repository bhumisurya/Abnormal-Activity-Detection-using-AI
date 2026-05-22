AI-Based Abnormal Activity Detection System

An intelligent real-time surveillance and human activity monitoring system developed using Machine Learning, MediaPipe Pose Estimation, OpenCV, Flask, and Firebase. The system detects suspicious or abnormal human activities from live webcam feeds and uploaded videos and automatically sends emergency alerts with user location details.

Project Overview

This project is designed to enhance security monitoring using Artificial Intelligence and Computer Vision. The application analyzes human body movements in real time using pose estimation techniques and classifies activities using a Machine Learning model.

The system extracts body landmarks from video frames using MediaPipe Pose, processes the pose coordinates, and predicts activities using a K-Nearest Neighbors (KNN) classifier trained on human movement patterns.

Whenever suspicious activity is detected, the system automatically:

Generates alerts
Sends email notifications
Shares user location details
Stores activity information securely using Firebase

The project supports:

Live webcam monitoring
Uploaded video analysis
Real-time activity detection
Cloud database integration
User authentication

Features:
Real-time human activity detection
AI-based abnormal behavior recognition
MediaPipe pose landmark extraction
Machine Learning activity classification using KNN
Live webcam feed analysis
Uploaded video processing
Email alert system with location details
Firebase authentication and cloud database
Flask-based web application
Multi-threaded video processing
Voice alert support
Activity dataset generation and training

Technologies Used:

Programming Language
Python
Libraries & Frameworks
OpenCV
MediaPipe
Flask
Scikit-learn
NumPy
Pandas
Threading
SMTP Email Automation
Database & Cloud
Firebase Authentication
Firebase Realtime Database

Machine Learning Workflow:

Capture video frames from webcam/video upload
Extract body pose landmarks using MediaPipe
Convert landmarks into feature vectors
Train KNN classifier on activity dataset
Predict human activity in real time
Detect suspicious/abnormal activities
Trigger alerts and notifications

Project Structure:

AI-Abnormal-Activity-Detection/
│
├── static/
├── templates/
├── dataset/
├── trained_model/
├── uploads/
├── app.py
├── train_model.py
├── requirements.txt
├── firebase_config.py
├── email_alert.py
└── README.md

Installation:

Clone the Repository
git clone https://github.com/your-username/AI-Abnormal-Activity-Detection.git
cd AI-Abnormal-Activity-Detection
Create Virtual Environment
python -m venv venv
Activate Environment
Windows
venv\Scripts\activate
Linux/Mac
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt

Run the Application:

python app.py

Open browser:

http://127.0.0.1:5000

Dataset & Training:

Human pose landmarks are collected using MediaPipe
Features are generated from body joint coordinates
KNN classifier is trained on activity patterns
Model is saved and reused for real-time prediction

Alert System

When abnormal activity is detected:

Email alerts are automatically triggered
User location details are included
Emergency notifications are sent instantly

Firebase Integration:

Firebase is used for:

User Authentication
Cloud Database Storage
User Activity Management
Real-time data synchronization

Future Improvements:

Deep Learning-based activity recognition
Mobile application integration
Face recognition support
Cloud deployment
Real-time CCTV integration
Advanced anomaly detection models

Applications:

Smart surveillance systems
Home security monitoring
Public safety systems
Elderly monitoring
Workplace safety analysis
