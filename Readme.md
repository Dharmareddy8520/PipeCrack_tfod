# Object Detection with TensorFlow

This project implements an object detection system using TensorFlow's Object Detection API. It leverages a pre-trained SSD MobileNet model, fine-tuned for custom object detection tasks such as detecting deformations, obstacles, ruptures, disconnects, misalignments, and depositions. The code supports training, evaluation, inference on images/videos, real-time detection, and model conversion for deployment (TFJS and TFLite).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Inference](#inference)
  - [Model Conversion](#model-conversion)
- [Exported Models](#exported-models)
- [Notes](#notes)

## Prerequisites

- Python 3.12+
- TensorFlow 2.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- TensorFlow Object Detection API
- `wget` for downloading pre-trained models
- Google Colab (optional, for cloud execution and Drive integration)

Install dependencies from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

For TensorFlow Object Detection API setup, follow the official guide: TensorFlow Object Detection Installation.("https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html")

Setup
Clone the TensorFlow Models Repository
git clone https://github.com/tensorflow/models.git Tensorflow/models

Download Pre-trained Model
The script downloads the SSD MobileNet V2 FPNLite 320x320 model from the TensorFlow Model Zoo:
http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

Prepare Dataset
Place training and testing images in Tensorflow/workspace/images.
Place videos in Tensorflow/workspace/vedios.
Generate TFRecords using generate_tfrecord.py

Directory Structure
The script defines paths automatically. Ensure your workspace matches this structure:

Tensorflow/
├── scripts/ # Utility scripts (e.g., generate_tfrecord.py)
├── models/ # TensorFlow models repository
├── protoc/ # Protocol buffers compiler
└── workspace/
├── annotations/ # Label map and TFRecords
├── images/ # Training/testing images
├── vedios/ # Videos for inference
├── models/ # Custom models and checkpoints
└── pre-trained-models/ # Downloaded pre-trained models

Key files:

label_map.pbtxt: Defines class labels.
pipeline.config: Model configuration for training/evaluation.
generate_tfrecord.py: Script to convert dataset to TFRecords (must be provided).
requirements.txt: Lists all Python dependencies.

Training the Model
Setup Paths and Download Pre-trained Model
Run the initial section to create directories and download the pre-trained model.
Create Label Map
The script generates label_map.pbtxt with the following classes:
Deformation (ID: 1)
Obstacle (ID: 2)
Rupture (ID: 3)
Disconnect (ID: 4)
Misalignment (ID: 5)
Deposition (ID: 6)

Generate TFRecords
Use generate_tfrecord.py to create train.record and test.record:

Update Configuration
The script updates pipeline.config for transfer learning with your dataset.
Train
Training is not explicitly scripted here. Use:

python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=10000

Inference
Image Detection
Detect objects in an image (e.g., 004557_2.jpg):
Output visualized with bounding boxes and labels.
Video Detection
Process a video (e.g., Chris Stevens - Google Chrome 2025-02-03 03-19-22.mp4):
Displays resized frames with detections (press q to exit).
Real-time Detection
Use a webcam for live detection:
Displays detections in an 800x600 window (press q to exit).

Convert to TFJS
Convert for TensorFlow.js:

tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores' --output_format=tfjs_graph_model --signature_name=serving_default Tensorflow/workspace/models/my_ssd_mobnet/export/saved_model Tensorflow/workspace/models/my_ssd_mobnet/tfjsexport

Convert to TFLite
Convert for TensorFlow Lite:

Exported Models
models.tar.gz: Zipped checkpoint directory for sharing/exporting.
