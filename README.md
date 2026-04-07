# Face Gesture Media Controller

Control Instagram Reels using **facial gestures** — no hands required.

---

## Features

* **Head Tilt Left** → Next Video
* **Head Tilt Right** → Previous Video
* **Mouth Open (Hold)** → Like

✔ Real-time face tracking using MediaPipe
✔ Noise-resistant gesture detection
✔ Anti-spam scroll protection
✔ Smooth and responsive interaction

---

## Tech Stack

* Python
* OpenCV
* MediaPipe Face Mesh
* PyAutoGUI

---

## Demo

![Demo](demo.gif)

> Will add later

---

## ⚙️ Installation

```bash
git clone https://github.com/Subhadeep-Dhar/face-gesture-media-controller.git
cd face-gesture-media-controller
pip install -r requirements.txt
python face_gesture_control.py
```

---

## How It Works

* Uses **MediaPipe Face Mesh (468 landmarks)**

* Detects:

  * Head tilt using eye alignment (slope)
  * Mouth open using lip distance ratio

* Applies:

  * Smoothing filters
  * Stability frames (anti-noise)
  * Cooldown system (anti-spam)
  * Yaw vs tilt separation (prevents false detection)

---

## ⚠️ Requirements

* Webcam
* Good lighting
* Keep browser window focused

---

## Challenges Solved

* False tilt detection due to head rotation (yaw)
* Spam scrolling from continuous tilt
* Noise from micro facial movements

✔ Implemented stability + filtering logic for reliable control

---

## Future Improvements

* Wink detection (save video)
* Gesture customization
* Cross-platform media control
* Blendshape-based detection (higher accuracy)

---

## Author

**Subhadeep Dhar**
