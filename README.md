# HKU-FYP25089-VARdict

### Pose Estimator: RTMPose
* YOLO has less dependency, but a simpler bone structure, no neck and glutes
* RTMPose: more points, more accurate structure compared to YOLO


### Environment
before all: export PYTHONNOUSERSITE=1, **IMPORTANT!!**
make sure in ~/.local/bin there is no pip*, in ~/.local/libs/python*/site-packages there are no packages.

torch==2.1.0
torchvision==0.16.0

imageio
sentencepiece
bitsandbytes==0.42.0 (for 8 bit quantization)
accelerate
protobuf

pip install -U openmim
mim install mmengine
install mmcv using the wheel file
mim install mmdet
mim install mmpose, requires chumpy, need to download and make a dummy one, also set --no-build-isolation

use flash-attn==2.5.7, compatible with torch 2.1.0, do not use pip install, use the wheel file in this repository

finally, use numpy==1.26.4, uninstall both opencv-python and opencv-python-headless, and then install opencv-python-headless. this is because mmpose and mmcv will automatically install the latest numpy and opencv-python, **IMPORTANT**: mmcv will reset opencv to opencv-python, not useable in server environment.

---

## 🌐 Remote Access Guide (HKU CS GPU Farm)

Since the demo runs on a compute node behind a gateway, you must use **SSH ProxyJump** to access the Web UI locally.

### 1. Get Compute Node IP

Inside your `gpu-interactive` session (where the demo is running), run:

```bash
hostname -I
# Example output: 10.21.XX.XX (Copy this IP)

```

### 2. Start SSH Tunnel

Run the following command on your **local computer** (do not run this on the server):

```bash
ssh -N -L 7860:<COMPUTE_NODE_IP>:7860 -J <USERNAME>@gpu2gate1.cs.hku.hk <USERNAME>@<COMPUTE_NODE_IP>

```

* **`<COMPUTE_NODE_IP>`**: The internal IP address you got from Step 1.
* **`<USERNAME>`**: Your student ID (e.g., `u3598820`).
* **`-N`**: This flag prevents opening a remote shell (port forwarding only).

### 3. Run the gradio script

use `python demo_launcher_0410.py --port 7860 --share` to start our VARdict demo.

### 4. Open in Browser

Once the tunnel is running, open your local browser and visit:
**[http://localhost:7860](https://www.google.com/search?q=http://localhost:7860)**

### Valid video format: .mp4

Although some files have a suffix of .mp4, it may be a fake .mp4 file. Please use the following command to cast the format if the .mp4 is not playable:  
`ffmpeg -i fake.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart -c:a aac real.mp4`



## Model

Our model uses Vicuna to process the input. We uses a feature projector to project the CLIP+Pose features (>1024dim) into the Vicuna embedding (1024dim). `load_model.py` loads our model, initialize with the X-VARS weights. If no projector weight is provided, we use random weights.

## Dataset
`dataset.py` defines our dataset, including the CLIP features and RTMPose features. There are two modes in this dataset, train and evaluate.
#### train mode
concatenate CLIP and Pose together.
Tokenize the annotations as the labels.

#### evaluate mode
returns the CLIP predicted features, Pose data, and two questions: "is this a foul? why?" and "what card should you give? why?".

## Training
`train.py` is our script for training. Our training contains two stages, first we freeze the Vicuna, and train the projector to project CLIP+Pose features consistently. The second stage involves LoRA fine-tuning. We fix the projector and fine-tune the Vicuna


## Evaluation
We use the evaluation mode in our dataset to conduct the evaluation. We focus on four metrics: foul accuracy, foul balanced accuracy, card accuracy and card balanced accuracy. We use ChatGPT to extract the natural language responses into labels. `evaluation_*_**.py` is our evaluation script.

## Results
We achieved a better result over foul accuracy and card accuracy plus balanced accuracy. But we did similar to X-VARS on foul balanced accuracy.

