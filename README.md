Description:
CamTensor recognises hand gestures (left, right or none) using a Raspberry Pi 3 (64-bit) and the Raspberry Pi Camera V2. A small C program reads frames from the camera, runs a TensorFlow Lite image model trained with Google Teachable Machine, and reports the detected gesture.

Required HW: Raspberrypi3-64 and camera Module V2 (IMX219).

Try: Use HDMI, after log in with root and try : journalctl -u gesture -f
