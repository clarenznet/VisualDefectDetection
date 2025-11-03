# Eye-Tracking Wearable Glasses with On-Device TinyML
## Complete Technical Implementation Guide

---

## 1. DETAILED ARCHITECTURE FOR ON-DEVICE TRAINING PIPELINE

### 1.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    WEARABLE GLASSES SYSTEM                   │
├─────────────────────────────────────────────────────────────┤
│  Hardware Layer                                              │
│  ├─ IR Eye Cameras (2x) → Eye Feature Extraction           │
│  ├─ Scene Camera (1x) → Field of View Capture              │
│  └─ IMU → Head Pose Estimation                              │
├─────────────────────────────────────────────────────────────┤
│  TinyML Processing Pipeline                                  │
│  ├─ Gaze Estimation Model (On-Device Inference + Training)  │
│  ├─ Object Detection Model (Inference Only)                 │
│  └─ Gaze-Object Fusion Engine                               │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                           │
│  └─ Context-Aware Assistance Services                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 On-Device Training Pipeline Details

#### Stage 1: Base Model Pre-Training (Offline)
```
Training Environment: Standard GPU/TPU
Dataset: Public gaze datasets (MPIIGaze, GazeCapture, etc.)
Model: Lightweight CNN for gaze regression

Architecture:
├─ Input: 64x64 eye region images
├─ Feature Extraction: MobileNetV2 backbone (quantization-aware training)
├─ Gaze Regression: Fully connected layers
└─ Output: 2D/3D gaze vector

Optimization:
├─ Quantization: INT8 weights, activations
├─ Pruning: 40-60% sparsity
├─ Knowledge Distillation: Teacher (large) → Student (tiny)
└─ Target Size: <200KB model, <50KB weights
```

#### Stage 2: On-Device Fine-Tuning Setup
```
Trigger: User performs calibration routine
Data Collection: 9-16 point calibration grid
Training Method: Transfer learning (freeze backbone, train head)

Memory Budget:
├─ Model Weights: 50KB
├─ Activations: 100KB (layer-by-layer processing)
├─ Gradients: 50KB (only for trainable layers)
├─ Training Data Buffer: 200KB
└─ Total: <512KB RAM
```

#### Stage 3: Lightweight Training Algorithm

**Option A: Backpropagation with Memory Optimization**
```python
# Pseudo-code for on-device training
class TinyGazeTrainer:
    def __init__(self):
        self.model = load_pretrained_model()
        self.freeze_backbone()
        
    def freeze_backbone(self):
        # Only train final layers
        for layer in self.model.layers[:-3]:
            layer.trainable = False
    
    def train_step(self, eye_image, target_gaze):
        # Forward pass with checkpointing
        features = self.model.backbone(eye_image)
        prediction = self.model.head(features)
        
        # Compute loss
        loss = mse_loss(prediction, target_gaze)
        
        # Backward pass (only for head layers)
        gradients = compute_gradients(loss, trainable_vars)
        
        # Update with small learning rate
        apply_gradients(gradients, lr=0.0001)
        
        return loss
    
    def online_learning(self, calibration_data):
        for epoch in range(5):  # Few epochs
            for eye_img, gaze in calibration_data:
                loss = self.train_step(eye_img, gaze)
```

**Option B: Forward-Forward Algorithm (More Memory Efficient)**
```python
# Alternative: No backpropagation needed
class ForwardForwardGaze:
    def train_layer(self, layer, positive_data, negative_data):
        # Positive pass
        pos_activity = layer(positive_data)
        pos_goodness = sum(pos_activity ** 2)
        
        # Negative pass (augmented incorrect gaze)
        neg_activity = layer(negative_data)
        neg_goodness = sum(neg_activity ** 2)
        
        # Update weights to maximize positive, minimize negative
        layer.weights += lr * (pos_goodness - neg_goodness)
```

#### Stage 4: Continual Learning Strategy
```
Online Adaptation:
├─ Replay Buffer: Store 100 most recent calibration samples
├─ Periodic Retraining: Every 1000 gaze estimates
├─ Drift Detection: Monitor prediction confidence
└─ Active Learning: Request calibration when confidence drops

Memory Management:
├─ Circular buffer for training data
├─ Low-rank adaptation (LoRA) for parameter-efficient fine-tuning
└─ Gradient checkpointing to reduce activation memory
```

---

## 2. SPECIFIC MODEL ARCHITECTURES FOR GAZE ESTIMATION

### 2.1 Architecture Comparison

| Model | Parameters | Size | Latency | Accuracy | On-Device Training |
|-------|-----------|------|---------|----------|-------------------|
| GazeNet-Tiny | 50K | 200KB | 15ms | ±4° | ✓ Fast |
| MobileGaze | 120K | 480KB | 25ms | ±3° | ✓ Moderate |
| EfficientGaze | 200K | 800KB | 40ms | ±2.5° | ✗ Slow |

### 2.2 Recommended: GazeNet-Tiny Architecture

```
INPUT: 64x64 grayscale eye image
    ↓
CONV1: 3x3, 16 filters, stride 2 → 32x32x16
    ↓
DEPTHWISE_CONV2: 3x3, stride 2 → 16x16x16
    ↓
POINTWISE_CONV3: 1x1, 32 filters → 16x16x32
    ↓
DEPTHWISE_CONV4: 3x3, stride 2 → 8x8x32
    ↓
POINTWISE_CONV5: 1x1, 64 filters → 8x8x64
    ↓
GLOBAL_AVG_POOL → 64 features
    ↓
FC1: 64 → 32 (trainable during fine-tuning)
    ↓
FC2: 32 → 2 (gaze_x, gaze_y) (trainable during fine-tuning)
    ↓
OUTPUT: 2D gaze vector

Total Parameters: ~48,000
Model Size: 192KB (INT8 quantized)
Inference Time: 12-18ms on Cortex-M7
```

### 2.3 Implementation Details

**Input Preprocessing:**
```python
def preprocess_eye_image(raw_image):
    # 1. Eye region detection (using lightweight Haar cascades or pre-computed ROI)
    eye_roi = extract_eye_region(raw_image)
    
    # 2. Resize to 64x64
    resized = resize(eye_roi, (64, 64))
    
    # 3. Normalize to [-1, 1]
    normalized = (resized / 127.5) - 1.0
    
    # 4. Apply histogram equalization (optional, for robustness)
    equalized = apply_clahe(normalized)
    
    return equalized
```

**Data Augmentation for Training:**
```python
augmentations = [
    RandomBrightness(±0.2),
    RandomContrast(±0.2),
    RandomRotation(±5°),
    RandomTranslation(±3 pixels),
    GaussianNoise(σ=0.01)
]
```

### 2.4 Advanced: 3D Gaze Estimation

For more accurate gaze-object mapping, estimate 3D gaze origin and direction:

```
INPUT: 64x64 eye image + head pose (from IMU)
    ↓
GAZENET-TINY (as above)
    ↓
FC_3D: 32 → 5 outputs
    ├─ gaze_theta (elevation angle)
    ├─ gaze_phi (azimuth angle)
    ├─ eye_origin_x
    ├─ eye_origin_y
    └─ eye_origin_z
    ↓
OUTPUT: 3D gaze ray in head coordinate system

Transform to world coordinates:
gaze_world = rotation_matrix(head_pose) @ gaze_head
```

---

## 3. HARDWARE RECOMMENDATIONS AND TRADE-OFFS

### 3.1 Microcontroller/SoC Comparison

#### **Option 1: STM32H747 (Dual-Core Cortex-M7/M4)**
```
Pros:
✓ 480 MHz Cortex-M7 with DSP instructions
✓ 1MB RAM, 2MB Flash
✓ Hardware acceleration for NN (X-CUBE-AI)
✓ Low power modes (idle: 50mA, active: 150mA)
✓ Excellent TensorFlow Lite Micro support

Cons:
✗ No on-chip camera interface (need external ISP)
✗ More complex development
✗ Higher cost ($10-15 per unit)

Best for: Research prototypes requiring maximum flexibility
```

#### **Option 2: ESP32-S3 (Dual-Core Xtensa LX7)**
```
Pros:
✓ 240 MHz dual-core with vector instructions
✓ 512KB RAM, 8MB PSRAM, 16MB Flash
✓ Built-in camera interface (DVP)
✓ WiFi/BLE for debugging/data collection
✓ Low cost ($3-5 per unit)
✓ Active TinyML community

Cons:
✗ Less efficient for NN inference vs Cortex-M7
✗ Higher power consumption (active: 200-300mA)
✗ Weaker FPU performance

Best for: Cost-sensitive prototypes, quick iteration
```

#### **Option 3: Arduino Nicla Vision**
```
Pros:
✓ STM32H747 + 2MP camera + TOF sensor
✓ Pre-integrated hardware, compact form factor
✓ Edge Impulse integration out-of-box
✓ Excellent for rapid prototyping

Cons:
✗ Fixed camera (less flexibility for eye tracking)
✗ Higher cost (~$100)
✗ Limited customization

Best for: Proof-of-concept, educational projects
```

#### **Option 4: Google Coral Dev Board Micro**
```
Pros:
✓ Cortex-M7 + Edge TPU (4 TOPS INT8)
✓ Ultra-fast inference (5ms for typical models)
✓ 64MB RAM
✓ Camera interface

Cons:
✗ Higher power consumption (TPU: 200-300mA)
✗ Overkill for simple gaze estimation
✗ More expensive (~$80)

Best for: Complex multi-model pipelines (gaze + object detection)
```

#### **Recommended Configuration: Hybrid Approach**
```
Primary MCU: STM32H747
├─ M7 core: Gaze estimation, fusion logic
└─ M4 core: Camera capture, preprocessing

Accelerator: Optional Edge TPU for scene understanding
Power Management: Dynamic voltage/frequency scaling
Battery: 500mAh LiPo → 6-8 hours operation

Estimated BOM Cost: $40-60 (prototype), $20-30 (production)
```

### 3.2 Camera Specifications

#### Eye-Tracking Cameras (2x)
```
Type: Near-infrared (NIR) 850nm
Sensor: OV7725 or similar (VGA 640x480)
Frame Rate: 60-90 FPS (higher = smoother tracking)
Interface: DVP (Digital Video Port)
IR Illumination: 850nm LED ring (invisible to user)
Power: 40mA per camera

Why NIR?
✓ Works in all lighting conditions
✓ Invisible to user (no distraction)
✓ Better pupil contrast
```

#### Scene Camera (1x)
```
Type: RGB color camera
Sensor: OV2640 or OV5640 (2-5MP)
Frame Rate: 15-30 FPS (lower to save power)
Field of View: 90-120° (match human peripheral vision)
Interface: DVP or MIPI CSI-2
Power: 80-120mA

Processing Strategy:
├─ Full resolution for object detection (every 500ms)
└─ Downsampled for real-time gaze projection (every 33ms)
```

### 3.3 Additional Sensors

```
IMU (Inertial Measurement Unit):
├─ Chip: BMI270 or LSM6DSO
├─ Purpose: Head pose estimation, motion compensation
├─ Sample Rate: 100 Hz
└─ Power: 0.5mA

Ambient Light Sensor (Optional):
├─ Chip: VEML7700
├─ Purpose: Adaptive IR illumination
└─ Power: 0.1mA

Time-of-Flight Sensor (Optional):
├─ Chip: VL53L1X
├─ Purpose: Depth estimation for 3D gaze
└─ Power: 20mA (only when active)
```

### 3.4 Power Budget & Battery Life

```
Component Power Breakdown (Typical):
├─ MCU (active inference): 150mA
├─ Eye cameras (2x): 80mA
├─ Scene camera: 100mA
├─ IR illumination: 40mA
├─ IMU: 0.5mA
├─ Peripherals: 30mA
└─ TOTAL: ~400mA @ 3.7V = 1.48W

Battery Options:
├─ 500mAh LiPo: 1.25 hours continuous
├─ 1000mAh LiPo: 2.5 hours continuous
└─ 2000mAh LiPo: 5 hours continuous

Power Optimization Strategies:
1. Dynamic frame rate: Reduce to 15 FPS during idle
2. Selective processing: Only run object detection on-demand
3. Sleep modes: 50% duty cycle → double battery life
4. IR dimming: Adaptive illumination based on pupil detection quality

TARGET: 8+ hours with 1500mAh battery (aggressive optimization)
```

---

## 4. METHODOLOGY FOR USER STUDIES

### 4.1 Study Design Framework

#### Phase 1: Technical Validation (Lab Study)
```
Participants: N=10-15 (diverse eye types, glasses wearers)
Duration: 60 minutes per session
Environment: Controlled lighting, static scenes

Metrics:
├─ Gaze Accuracy: Angular error (degrees)
├─ Calibration Efficiency: Time to achieve <3° error
├─ On-Device Training: Convergence speed, memory usage
├─ System Latency: End-to-end delay
└─ Battery Life: Runtime under continuous use

Protocol:
1. Initial calibration (9-point grid)
2. Accuracy test (25 random targets)
3. On-device training session
4. Re-test accuracy (measure improvement)
5. Long-duration tracking (30 min)
6. Repeat after fatigue/position shift
```

#### Phase 2: Application-Specific Study (Real-World)
```
Participants: N=20-30 (target user population)
Duration: 2-week deployment + follow-ups
Environment: Natural settings (home, work, outdoors)

Use Case Examples:
A. Reading Assistance
   ├─ Task: Read articles, books with gaze-activated TTS
   ├─ Metrics: Reading speed, comprehension, user preference
   
B. Object Recognition
   ├─ Task: Identify 20 household objects via gaze
   ├─ Metrics: Recognition accuracy, lookup time, utility rating

C. Navigation Aid (for visually impaired)
   ├─ Task: Navigate indoor/outdoor routes
   ├─ Metrics: Obstacle avoidance, destination accuracy, confidence

Data Collection:
├─ Quantitative: Accuracy logs, interaction logs, battery data
├─ Qualitative: Semi-structured interviews, diary studies
└─ Physiological: Eye strain (NASA-TLX), comfort ratings
```

### 4.2 Evaluation Metrics

#### Technical Metrics
```
1. Gaze Accuracy
   Formula: Angular error = arccos(gaze_predicted · gaze_true)
   Target: <3° (functional), <2° (excellent)

2. Precision (Jitter)
   Formula: σ = std(gaze_samples) over 1 second fixation
   Target: <0.5° RMS

3. Calibration Efficiency
   ├─ Initial: Points needed for <3° accuracy
   ├─ Adaptation: Accuracy improvement after 100 predictions
   Target: <16 points initial, >20% improvement

4. System Latency
   ├─ Eye capture to gaze output: <50ms
   └─ Gaze to object identification: <200ms
   
5. Power Efficiency
   Formula: mAh per hour of active tracking
   Target: <180mAh/hr (→ 8hr on 1500mAh)
```

#### User Experience Metrics
```
1. NASA Task Load Index (TLX)
   ├─ Mental Demand
   ├─ Physical Demand
   ├─ Temporal Demand
   ├─ Performance
   ├─ Effort
   └─ Frustration
   Target: <50/100 (low workload)

2. System Usability Scale (SUS)
   10-item questionnaire, score 0-100
   Target: >68 (above average), >80 (excellent)

3. User Acceptance
   ├─ Perceived Usefulness (TAM model)
   ├─ Perceived Ease of Use
   ├─ Intention to Use
   └─ Privacy Concerns
   5-point Likert scales

4. Wearability Comfort
   ├─ Physical comfort (weight, pressure points)
   ├─ Visual comfort (no obstruction)
   ├─ Social acceptability
   └─ Wearing duration tolerance
```

### 4.3 Experimental Protocols

#### Calibration Study Protocol
```
Objective: Compare calibration methods for on-device training

Conditions (within-subjects):
A. Traditional 9-point grid
B. 16-point extended grid
C. Smooth pursuit (follow moving target)
D. Natural viewing (watch video, auto-label)

Procedure per condition:
1. Participant performs calibration (record time)
2. Test accuracy on 25 random targets
3. Compute angular error statistics
4. Participant rates difficulty (1-5 scale)

Analysis:
├─ ANOVA: Accuracy by calibration method
├─ Correlation: Calibration time vs. accuracy
└─ User preference ranking
```

#### Adaptation Study Protocol
```
Objective: Evaluate continual learning over time

Design: Longitudinal (7 days)
Day 1: Initial calibration + baseline accuracy
Day 2-6: Normal use with periodic testing
Day 7: Final accuracy test + recalibration

Tracked Variables:
├─ Accuracy drift over time
├─ Number of auto-corrections triggered
├─ Model update frequency
└─ User intervention rate

Hypotheses:
H1: Accuracy improves with usage (continual learning)
H2: Drift is minimal with periodic updates
H3: User corrections reduce over time
```

#### Application Effectiveness Study
```
Use Case: Reading assistance for low-vision users

Participants: N=15 (visual acuity 20/200 to 20/70)

Task: Read 3 articles (500 words each)
Conditions (counterbalanced):
A. Baseline: Smartphone magnifier + TTS
B. Prototype: Gaze-activated TTS (look to read)
C. Control: Human assistant reads aloud

Metrics:
├─ Reading speed (words per minute)
├─ Comprehension (5 questions per article)
├─ Subjective preference (ranking)
└─ Eye strain (0-10 scale)

Statistical Test: Repeated measures ANOVA
```

### 4.4 Ethical Considerations & IRB Protocol

```
Required Approvals:
├─ Institutional Review Board (IRB) for human subjects
├─ Informed consent (video recording, biometric data)
└─ Data privacy plan (GDPR/HIPAA if applicable)

Privacy Protections:
├─ All processing on-device (no cloud uploads)
├─ Eye images deleted after processing
├─ Gaze data anonymized
├─ Participant right to delete data
└─ Secure storage for study data

Risk Mitigation:
├─ Eye strain: Limit sessions to 45 min, breaks every 15 min
├─ Dizziness: Screen for motion sensitivity, allow opt-out
├─ Privacy: Clear disclosure of data collection
└─ Inclusivity: Test with diverse populations (age, ethnicity, vision abilities)

Informed Consent Elements:
1. Purpose of research
2. Procedures and time commitment
3. Risks and discomforts (minimal: eye strain)
4. Benefits (may not benefit directly, advances assistive tech)
5. Confidentiality measures
6. Right to withdraw
7. Contact information
```

---

## 5. IMPLEMENTATION ROADMAP (DETAILED)

### Timeline: 18-Month Research Project

#### **Months 1-3: Foundation & Literature Review**

**Week 1-4: Literature Review**
```
Topics to cover:
├─ Gaze estimation: CNN-based, appearance-based, hybrid methods
├─ TinyML: Model compression, quantization, on-device training
├─ Wearable systems: Power optimization, form factors
└─ Assistive technology: User needs, evaluation frameworks

Key Papers (starting points):
├─ "GazeCapture: Large-Scale Dataset" (MIT, 2016)
├─ "It's Written All Over Your Face" (Google, 2018)
├─ "TinyML: Machine Learning with TensorFlow Lite" (O'Reilly, 2019)
└─ "Federated Learning for Mobile Devices" (Google, 2017)

Deliverable: Literature review document (20-30 pages)
```

**Week 5-8: Dataset Preparation**
```
Actions:
1. Download public datasets:
   ├─ MPIIGaze: 213,000 images, 15 participants
   ├─ GazeCapture: 2.5M frames, 1,474 participants
   └─ RT-GENE: 122,000 images with 3D gaze

2. Preprocess for TinyML:
   ├─ Resize to 64x64
   ├─ Apply augmentation pipeline
   ├─ Split: 80% train, 10% val, 10% test
   └─ Export to TFRecord format

3. Train baseline models:
   ├─ Full-size model (for comparison)
   ├─ Compressed model (target for deployment)
   └─ Benchmark accuracy vs. size trade-offs

Deliverable: Pre-trained models, performance report
```

**Week 9-12: Hardware Procurement & Testing**
```
Purchase:
├─ 3x STM32H747 dev boards ($150)
├─ 6x OV7725 NIR cameras ($60)
├─ 3x OV2640 RGB cameras ($30)
├─ IMU modules, power supplies, etc. ($100)
├─ 3D printer time for frame prototypes ($50)
└─ Total: ~$400

Initial tests:
├─ Camera interfacing (DVP protocol)
├─ IMU calibration
├─ Power consumption baseline
└─ TFLite Micro benchmark (inference speed)

Deliverable: Working camera capture system
```

---

#### **Months 4-7: Hardware Prototype & Model Porting**

**Week 13-16: Mechanical Design**
```
Prototype v1: Breadboard setup
├─ Mount cameras on adjustable rig
├─ Test eye-tracking positioning
└─ Validate field-of-view alignment

Prototype v2: 3D-printed frame
├─ Design in Fusion 360 / SolidWorks
├─ Print frame (multiple iterations expected)
├─ Integrate electronics
└─ Balance weight distribution (<100g target)

Deliverable: Wearable prototype (functional, not aesthetic)
```

**Week 17-22: Model Optimization & Deployment**
```
Step 1: Quantization-Aware Training (QAT)
├─ Retrain model with fake quantization nodes
├─ Target: INT8 weights and activations
└─ Benchmark: Accuracy drop <0.5°

Step 2: Post-Training Quantization
├─ Convert TensorFlow model → TFLite
├─ Apply dynamic range quantization
└─ Validate accuracy on test set

Step 3: Deployment to MCU
├─ Port to TFLite Micro
├─ Optimize with CMSIS-NN kernels
├─ Profile memory and latency
└─ Achieve <50ms inference

Step 4: Integration Testing
├─ Full pipeline: Camera → Preprocessing → Inference
├─ Validate real-time performance (30+ FPS)
└─ Stress test: 1-hour continuous operation

Deliverable: Real-time gaze estimation on prototype
```

**Week 23-28: Scene Understanding Model**
```
Object Detection Options:
A. YOLO-Nano (fastest, lower accuracy)
B. MobileNetV2 + SSD (balanced)
C. EfficientDet-Lite0 (most accurate)

Selected: MobileNetV2-SSDLite (COCO dataset)
├─ Pre-trained weights from TF Model Zoo
├─ Quantize to INT8
├─ Deploy to second MCU or time-share with gaze model

Integration:
├─ Run object detection at 2 Hz (every 500ms)
├─ Cache detections for real-time gaze projection
└─ Optimize: Only detect in gaze vicinity (ROI)

Deliverable: Gaze + object detection working together
```

---

#### **Months 8-11: On-Device Training Implementation**

**Week 29-34: Training Algorithm Development**
```
Approach: Transfer learning with frozen backbone

Implementation Steps:
1. Identify trainable layers:
   ├─ Freeze: All convolutional layers
   └─ Train: Final 2 fully-connected layers (~5K params)

2. Implement backpropagation for FC layers:
   ├─ Forward pass: Use TFLite Micro
   ├─ Backward pass: Custom gradient computation
   └─ Optimizer: SGD with momentum (lightweight)

3. Memory optimization:
   ├─ In-place gradient updates
   ├─ Single-sample mini-batches
   └─ Gradient accumulation if needed

4. Calibration data collection:
   ├─ Display calibration points on external screen
   ├─ User looks at each point for 2 seconds
   ├─ Collect 10 samples per point
   └─ Total: 90-160 training samples

5. Training loop:
   for epoch in 1..5:
       for sample in calibration_data:
           loss = forward_backward(sample)
           update_weights()
       validate()
   
Benchmarks:
├─ Training time: <30 seconds for 5 epochs
├─ Memory peak: <512KB RAM
├─ Accuracy improvement: >20% reduction in angular error

Deliverable: Working on-device training system
```

**Week 35-40: Continual Learning & Adaptation**
```
Implement:
1. Drift detection:
   ├─ Monitor prediction confidence (softmax entropy)
   ├─ If confidence < threshold → trigger recalibration
   
2. Replay buffer:
   ├─ Store 100 recent high-confidence predictions
   ├─ Mix with new calibration data (80% old, 20% new)
   └─ Prevents catastrophic forgetting

3. Active learning:
   ├─ Identify uncertain regions in gaze space
   ├─ Request user to look at those areas
   └─ Prioritize samples for training

4. Periodic retraining:
   ├─ Every 1000 predictions OR
   ├─ When user manually requests OR
   ├─ When drift detected

Test over 1 week of simulated use:
├─ Day 1: Initial calibration
├─ Day 3: Introduce drift (change glasses position)
├─ Day 5: Automatic adaptation
└─ Day 7: Evaluate final accuracy

Deliverable: Robust long-term tracking system
```

---

#### **Months 12-15: Application Development & User Testing**

**Week 41-46: Assistance Features**
```
Feature 1: Text-to-Speech Reader
├─ Integrate Tesseract OCR (lightweight, on-device)
├─ Gaze dwells on text → highlight + read aloud
├─ Use Pico TTS or send to phone via BLE
└─ Optimize: Only OCR gaze region (not full frame)

Feature 2: Object Identification
├─ User looks at object for 2 seconds
├─ Object detection + Google Lens API (optional cloud)
├─ Speak object name + relevant info
└─ Cache common objects for offline use

Feature 3: Navigation Assistance
├─ Depth estimation (ToF sensor or monocular depth CNN)
├─ Obstacle detection in gaze path
├─ Audio cues: "Obstacle ahead, turn left"
└─ Semantic mapping (doorways, stairs, etc.)

Deliverable: 3 working demo applications
```

**Week 47-52: Pilot User Study (Phase 1)**
```
Participants: N=10 (5 sighted, 5 low-vision)

Protocol (per participant):
Day 1: Introduction, calibration, training (1 hour)
Day 2-6: Take-home usage (30 min/day, logged)
Day 7: In-lab testing, interview (1 hour)

Tasks:
├─ Controlled accuracy test (lab)
├─ Reading task (3 articles with TTS)
├─ Object identification (10 household items)
└─ Navigation (indoor obstacle course)

Data Collected:
├─ Quantitative: Accuracy, task completion time
├─ Qualitative: Interview transcripts, feedback
├─ System logs: Battery life, errors, model updates

Analysis:
├─ Descriptive statistics (accuracy, usability)
├─ Thematic analysis of interviews
└─ Identify pain points for iteration

Deliverable: Pilot study report, system refinements
```

---

#### **Months 16-18: Full Study & Publication**

**Week 53-60: Full User Study (Phase 2)**
```
Participants: N=30 (15 per use case group)

Group A: Reading assistance for low-vision
Group B: Memory assistance for mild cognitive impairment

Extended protocol (2 weeks take-home):
├─ Baseline (Week 0): Pre-tests without system
├─ Training (Week 1): Intensive use, daily logs
├─ Evaluation (Week 2): Post-tests, interviews
└─ Follow-up (Week 4): Long-term retention

Measures:
├─ Primary: Task performance improvement
├─ Secondary: User acceptance, workload
└─ Exploratory: Behavioral changes, independence

Statistical Analysis:
├─ Mixed ANOVA: Group × Time interaction
├─ Regression: Predict outcomes from usage patterns
└