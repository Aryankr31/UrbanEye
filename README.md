# 🌆 UrbanEYE
### **AI-powered Smart Surveillance & Traffic Management System**

---

## 📖 **Overview**
**UrbanEYE** is an **AI-based system** that transforms existing CCTV cameras into intelligent surveillance tools for smart cities. It enhances **traffic management, public safety, and urban mobility** by combining **computer vision, deep learning, and real-time analytics**.

### ✅ **Key Capabilities:**
- 🚦 **Adaptive Traffic Light Control** – Dynamically adjusts green light duration using vehicle density.
- 🚑 **Accident Detection** – Detects crashes in real-time and alerts authorities immediately.
- 👥 **Crowd Sentiment Analysis** – Identifies aggression, panic, or unusual behavior to prevent riots.
- 🔎 **Automatic Number Plate Recognition (ANPR)** – Detects and logs vehicle license plates for violations & monitoring.
- 📡 **Scalable Deployment** – Works with existing CCTV infrastructure → **low-cost & scalable** solution.

---

## 🏗️ **Project Architecture**

**UrbanEYE's architecture** is a streamlined pipeline for processing video data and generating actionable insights.

- **Input Layer**: Ingests video feeds from **live CCTV cameras** or pre-recorded video files.
- **Detection Module**: Utilizes the high-performance **YOLOv8 model** for accurate and fast detection of vehicles, pedestrians, and other relevant objects.
- **Tracking Module**: Employs industry-standard multi-object tracking algorithms, **DeepSORT or ByteTrack**, to maintain persistent IDs for detected objects across video frames.
- **Decision Layer**: Processes tracking data to make intelligent decisions for various applications, such as adjusting traffic signals, triggering emergency alerts, or performing sentiment analysis.
- **Output Layer**: Provides a user-friendly interface for visualizing data, including **dashboards, real-time alerts**, and integration with city-wide traffic management systems.

```

Input Layer        →  Live CCTV feeds / recorded videos
Detection Module   →  YOLOv8 for vehicle, person & object detection
Tracking Module    →  DeepSORT / ByteTrack for real-time tracking
Decision Layer     →  Traffic → Adaptive signals
Accidents → Emergency alerts
Crowd → Sentiment analysis
Output Layer       →  Dashboards, alerts, or direct traffic system integration

```

---

## 📂 **Repository Structure**

The project repository is organized into a clean, logical structure to facilitate development and collaboration.

```

UrbanEYE/
│── data/                       \# Stores sample datasets and video files for testing and development
│── models/                     \# Contains pre-trained and fine-tuned AI model weights
│── outputs/                    \# Directory for storing processed outputs, log files, and analysis results
│── src/                        \# The core source code for all project modules
│   ├── traffic\_control.py       \# Implements the adaptive traffic signal logic
│   ├── accident\_detection.py    \# Contains the pipeline for accident detection
│   ├── sentiment\_analysis.py    \# Code for analyzing crowd sentiment
│   └── main.py                  \# The central entry point for running the application
│── requirements.txt             \# Lists all Python dependencies required to run the project
│── .gitignore                   \# Specifies files and directories that Git should ignore (e.g., temporary files, outputs)
│── README.md                    \# This comprehensive documentation file

````

---

## 🚀 **Installation & Setup**

To get a local copy of UrbanEYE up and running, follow these simple steps.

### **1️⃣ Clone the Repository**
Begin by cloning the project from GitHub using your terminal.

```bash
git clone [https://github.com/Aryankr31/UrbanEye.git](https://github.com/Aryankr31/UrbanEye.git)
cd UrbanEye
````

### **2️⃣ Install Dependencies**

Install all necessary Python libraries from the **`requirements.txt`** file. It's recommended to use a **virtual environment**.

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Application**

Execute the **`main.py`** file to start the UrbanEYE application.

```bash
python src/main.py
```

-----

## 🛠️ **Tech Stack**

**UrbanEYE** is built on a modern, robust tech stack designed for performance and scalability.

  - **Python** – The primary programming language used for development.
  - **YOLOv8 (Ultralytics)** – A state-of-the-art model for **real-time object detection**.
  - **DeepSORT / ByteTrack** – Advanced algorithms for **multi-object tracking**.
  - **OpenCV** – A powerful library for all **video processing** and computer vision tasks.
  - **PyTorch** – The **deep learning framework** used for model inference.
  - **Streamlit** – For building **interactive dashboards** and visualizations.

-----

## 🌍 **Real-World Impact**

**UrbanEYE** has the potential to significantly improve the quality of life in cities by:

  - **⚡ Reducing urban traffic congestion** by optimizing traffic flow and decreasing travel times.
  - **🚑 Improving emergency response times** by providing instant alerts for accidents.
  - **🛡️ Enhancing public safety** by proactively identifying and responding to potentially dangerous situations.
  - **💰 Providing a cost-efficient solution** by reusing existing CCTV infrastructure, saving cities significant funds on new hardware.

-----

## 📌 **Future Scope**

We have several plans to expand **UrbanEYE's capabilities**:

  - **🔗 IoT Integration** – Implement direct communication protocols for seamless control of smart traffic lights.
  - **🖥️ Edge Deployment** – Optimize the model for resource-constrained devices like **Jetson Nano and Raspberry Pi** for on-site processing.
  - **🌐 City-wide Integration** – Develop a centralized API for integration with broader city management platforms.
  - **🎥 Multi-camera Tracking** – Enhance the tracking module to follow vehicles and individuals across multiple intersections.

-----

## ✨ **Acknowledgements**

A huge thank you to the creators and maintainers of the following open-source projects that made **UrbanEYE** possible:

  - **Ultralytics YOLOv8**
  - **DeepSORT**
  - **ByteTrack**
  - The entire open-source **computer vision and machine learning community** ❤️

<!-- end list -->

```
```
