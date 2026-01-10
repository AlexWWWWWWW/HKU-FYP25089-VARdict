# HKU-FYP25089-VARdict

### Pose Estimator: YOLO vs OpenPose
* YOLO has less dependency, but a simpler bone structure, no neck and glutes
* MMPose: more points, more accurate structure compared to YOLO


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

use flash-attn==2.8.3, the latest that handles dependencies automatically. use --no-build-isolation --no-cache-dir

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

such as`python gradio_script.py --port 7860`

### 4. Open in Browser

Once the tunnel is running, open your local browser and visit:
**[http://localhost:7860](https://www.google.com/search?q=http://localhost:7860)**


