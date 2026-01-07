# HKU-FYP25089-VARdict

### Pose Estimator: YOLO vs OpenPose
* YOLO has less dependency, but a simpler bone structure, no neck and glutes
* MMPose: more points, more accurate structure compared to YOLO


### Environment
before all: export PYTHONNOUSERSITE=1, **IMPORTANT!!**
make sure in ~/.local/bin there is no pip*, in ~/.local/libs/python*/site-packages there are no packages.

torch==2.1.0
torchvision==0.16.0

pip install -U openmim
mim install mmengine
install mmcv using the wheel file
mim install mmdet
mim install mmpose, requires chumpy, need to download and make a dummy one, also set --no-build-isolation

use flash-attn==2.8.3, the latest that handles dependencies automatically. use --no-build-isolation --no-cache-dir

finally, use numpy==1.26.4, uninstall both opencv-python and opencv-python-headless, and then install opencv-python-headless. this is because mmpose and mmcv will automatically install the latest numpy and opencv-python, **IMPORTANT**: mmcv will reset opencv to opencv-python, not useable in server environment.


