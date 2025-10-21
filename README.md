# VisualDefectDetection
This project uses a raspberry pi 5 to perform assembly line monitoring, statistics collection and recording. This includes drawing bounding boxes to highlight misaligned components during assembly, counting the number of detected errors, collecting statistics on assembled devices, using a second station to determine tardiness on the assembly line.

# Prototyping Architecture

# Hardware:

### Raspberry Pi 5 with active cooling fan

### Pi Camera Module 2 

### Diffused ring LED

### External SSD for data logging

# Software Stack:

### OS: Raspberry Pi OS (Bookworm)

### Libraries: OpenCV, TensorFlow Lite / YOLOv8n (Ultralytics)

### Real-time dashboard: Flask + MQTT + Grafana (optional)

### Edge decision logic: “OK / NG” binary output + indicator light + alarm + display error highlight + notitication broadcast.
