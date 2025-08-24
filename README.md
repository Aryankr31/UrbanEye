🌆 UrbanEYE
AI-powered Smart Surveillance & Traffic Management System

UrbanEYE is an AI-based system that transforms existing CCTV cameras into smart surveillance tools for cities.
It optimizes traffic light control, detects accidents in real-time, performs crowd sentiment analysis for riot prevention, and supports license plate recognition — making urban mobility safer, faster, and smarter.

🚀 Features
Adaptive Traffic Light Control – Uses vehicle detection to adjust green light duration dynamically.
Accident Detection – Identifies crashes and alerts authorities.
Crowd Sentiment Analysis – Detects aggression, panic, or unusual behavior to prevent riots.
License Plate Recognition (ANPR) – Reads and logs vehicle numbers for violations & safety monitoring.
Scalable – Works with existing CCTV infrastructure.

🏗️ Project Architecture -
Input Layer – Live CCTV feed or recorded video.
Detection Module – YOLOv8 for vehicle/person/object detection.
Tracking Module – DeepSORT & ByteTrack for real-time object tracking.
Decision Layer – Traffic flow → Adaptive signal control.
                 Accidents → Immediate alert generation. 
                 Crowd → Emotion/sentiment detection.
Output Layer – Dashboards, alerts, or direct integration with traffic systems.

📂 Repository Structure
UrbanEYE/
│── data/              
│── models/            
│── outputs/           
│── src/               
│   ├── traffic_control.py
│   ├── accident_detection.py
│   ├── sentiment_analysis.py
│   └── main.py
│── requirements.txt
│── .gitignore
│── README.md


⬇️ Install dependencies:
pip install -r requirements.txt


🛠️ Tech Stack
Python
YOLOv8 (Ultralytics) – Object detection
DeepSORT – Multi-object tracking
OpenCV – Video processing
PyTorch – Model inference
Streamlit – Dashboard visualization

🌍 Real-World Impact
Reduce urban traffic congestion.
Improve emergency response times.
Enhance public safety in riot-prone areas.
Reuse existing CCTV infrastructure → low-cost & scalable.

📌 Future Scope
Integration with IoT traffic lights for direct signal control.
Deployment on edge devices (Jetson Nano, Raspberry Pi).
Integration with city-wide traffic management systems.
Add multi-camera tracking across intersections.
