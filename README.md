# 🚀 TurboThrill Frame AI v2.0

> **An AI-powered Computer Vision application that automatically detects and extracts high-quality motorcycle spark frames from videos using OpenCV and Streamlit.**

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-v2.0-success?style=for-the-badge)

---

## 🎥 Demo

> 📹 **Project Demo:** *(Add your demo video/GIF here)*

> 🌐 **Live Demo:** *(Add your Streamlit deployment link here)*

---

# 📖 Overview

TurboThrill Frame AI is an AI-powered Computer Vision application that automatically detects and extracts motorcycle spark moments from videos.

Instead of manually reviewing thousands of video frames, the system intelligently identifies spark events using OpenCV-based image processing techniques and exports only the most relevant frames with detailed metadata.

The project demonstrates practical Computer Vision engineering by combining modular software architecture with an interactive Streamlit interface.

---

# ✨ Features

- 🔥 Automatic Spark Detection
- 🎯 Confidence-based Detection
- 🌈 Dual HSV Color Filtering
- 💡 Brightness Filtering
- 🧹 Morphological Noise Removal
- 📸 Duplicate Frame Elimination
- 📊 Metadata Export (JSON)
- 📄 CSV Report Generation
- 📂 Batch Folder Processing
- ⚙️ YAML Configuration
- 📈 Processing Statistics
- 🖥️ Interactive Streamlit Dashboard

---

# 🏗️ Project Structure

```text
TurboThrill-Frame-AI/
│
├── dashboard/
│   ├── app.py
│   └── styles.py
│
├── detectors/
│   └── spark_detector.py
│
├── processor/
│   ├── folder_processor.py
│   ├── video_processor.py
│   └── duplicate_remover.py
│
├── exporters/
│   └── metadata_exporter.py
│
├── config/
│   └── default_config.yaml
│
├── uploads/
├── outputs/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ Detection Pipeline

```text
Video Input
      │
      ▼
Frame Extraction
      │
      ▼
Brightness Filtering
      │
      ▼
HSV Color Conversion
      │
      ▼
Dual Color Mask
      │
      ▼
Morphological Cleanup
      │
      ▼
Contour Detection
      │
      ▼
Area Filtering
      │
      ▼
Confidence Scoring
      │
      ▼
Duplicate Removal
      │
      ▼
Save Best Frames
      │
      ▼
Export Metadata & Reports
```

---

# 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core Programming |
| OpenCV | Computer Vision |
| NumPy | Image Processing |
| Streamlit | Interactive Dashboard |
| Pandas | Report Generation |
| YAML | Configurable Parameters |
| JSON | Metadata Storage |

---

# 📂 Output Structure

```text
outputs/

Video_1/

    spark_001.jpg
    spark_002.jpg
    spark_003.jpg

    metadata.json
    report.csv

Video_2/

    spark_001.jpg
    spark_002.jpg

    metadata.json
    report.csv
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/TurboThrill-Frame-AI.git
```

## Navigate

```bash
cd TurboThrill-Frame-AI
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run

```bash
streamlit run dashboard/app.py
```

---

# ⚙️ Configuration

All detection parameters are stored inside:

```text
config/default_config.yaml
```

You can modify:

- HSV color ranges
- Brightness threshold
- Minimum spark area
- Confidence threshold
- Frame sampling interval

without changing the source code.

---

# 📊 Generated Outputs

For every processed video, the application generates:

### Extracted Frames

```text
spark_001.jpg
spark_002.jpg
...
```

### Metadata

```text
metadata.json
```

Contains:

- Frame Number
- Timestamp
- Confidence Score
- Spark Area
- Resolution

### Report

```text
report.csv
```

Contains processing statistics and detection summaries.

---

# 🎯 Current Capabilities

- ✅ Detects motorcycle spark frames
- ✅ Eliminates duplicate frames
- ✅ Processes entire folders automatically
- ✅ Exports structured metadata
- ✅ Generates CSV reports
- ✅ Configurable detection engine
- ✅ Interactive Streamlit interface

---

# 🔮 Roadmap

- 🤖 Deep Learning-based Spark Detection
- 🚀 YOLO Integration
- ⚡ GPU Acceleration
- 📹 Video Preview
- 📈 Spark Analytics Dashboard
- ☁️ Cloud Storage
- 🐳 Docker Support
- 🌐 REST API
- 🎯 Automatic Best Frame Ranking

---

# 📸 Screenshots

## Dashboard

> *(Add Screenshot)*

---

## Detection Results

> *(Add Screenshot)*

---

## Output Folder

> *(Add Screenshot)*

---

# 💡 Why This Project?

Finding the perfect spark moment in motorcycle videos is a tedious manual process.

TurboThrill Frame AI automates this workflow by combining Computer Vision techniques with a modular processing pipeline, making spark extraction significantly faster and more reliable.

Beyond solving a real-world problem, the project also demonstrates software engineering principles such as modular architecture, configurable pipelines, metadata generation, and scalable project organization.

---

# 🤝 Contributing

Contributions are always welcome.

1. Fork the repository
2. Create your feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## Konain Fatima

**AI/ML Engineer • Computer Vision Enthusiast • B.Tech AI & ML**

- GitHub: https://github.com/yourusername
- LinkedIn: https://linkedin.com/in/yourprofile

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub. It helps others discover the project and supports future development.
