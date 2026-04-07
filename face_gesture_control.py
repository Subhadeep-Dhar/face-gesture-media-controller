"""
Reels & Shorts Gesture Controller
===================================
Controls YouTube Shorts / Instagram Reels using facial gestures.
Uses ONLY keyboard shortcuts — no mouse movement at all.

HOW TO USE:
  1. Run this script in VS Code terminal
  2. Open Chrome → go to YouTube Shorts or Instagram Reels
  3. Click ONCE inside the Chrome window to give it focus
  4. Come back and do gestures — they fire into Chrome automatically
  5. The small webcam window runs in the corner, never steals focus

GESTURES:
  Smile (hold 1s)       -> Like current video
  Eyebrows UP (hold 1s) -> Next video / scroll down
  Mouth OPEN (hold 1s)  -> Previous video / scroll up

MODE SWITCHING (press key while webcam window is active):
  Press 'y' -> YouTube Shorts mode  (Down/Up/L)
  Press 'i' -> Instagram Reels mode (Down/Up/L)
  Press 'q' -> Quit

KEYBOARD SHORTCUTS USED:
  YouTube Shorts:   Down Arrow = next,  Up Arrow = previous,  L = like
  Instagram Reels:  Down Arrow = next, Up Arrow = previous,  L = like (may not work, no official shortcut)

Install:
  pip install mediapipe==0.10.14 opencv-python pyautogui numpy==1.26.4

Run:
  python reels_gesture_control.py
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import collections

# ─────────────────────────────────────────────
# SENSITIVITY THRESHOLDS — tune if needed
# ─────────────────────────────────────────────
SMILE_THRESHOLD      = 0.50   # mouth width / face width ratio
BROW_RAISE_THRESHOLD = 0.05   # adjusted for outer brow
MOUTH_OPEN_THRESHOLD = 0.10   # mouth height / face height ratio for open mouth
REQUIRED_HOLD_FRAMES = 5     # frames gesture must be held (~1.3 sec at 15fps) 20
COOLDOWN_SECONDS     = 2.5    # seconds between repeated triggers
SMOOTH_WINDOW        = 15     # landmark position smoothing window


# ─────────────────────────────────────────────
# MediaPipe — tasks API with face_landmarker.task
# ─────────────────────────────────────────────
# Landmark indices
LM_MOUTH_LEFT   = 61
LM_MOUTH_RIGHT  = 291
LM_MOUTH_TOP    = 13
LM_MOUTH_BOTTOM = 14
LM_LBROW = [276, 283, 282]   # left eyebrow cluster
LM_RBROW = [46, 53, 52]      # right eyebrow cluster
LM_LEYE_TOP     = 386  # Top of left eye
LM_REYE_TOP     = 159  # Top of right eye
LM_FOREHEAD     = 10 
LM_CHIN         = 152
LM_LEFT_EAR     = 234
LM_RIGHT_EAR    = 454


brow_baseline = None
baseline_frames = []
CALIBRATION_FRAMES = 30
is_calibrated = False

prev_face_center_y = None

prev_nod_center_y = None

last_nod_time = 0

tilt_frames = 0




def avg_y(landmarks, indices, w, h):
    ys = []
    for idx in indices:
        _, y = get_smoothed_landmark(landmarks, idx, w, h)
        ys.append(y)
    return sum(ys) / len(ys)

# ─────────────────────────────────────────────
# Landmark smoothing
# ─────────────────────────────────────────────
smooth_buffers = {}


def get_smoothed_landmark(landmarks, idx, w, h):
    """Smooth landmark position over last SMOOTH_WINDOW frames."""
    lm = landmarks[idx]
    x, y = lm.x * w, lm.y * h
    if idx not in smooth_buffers:
        smooth_buffers[idx] = collections.deque(maxlen=SMOOTH_WINDOW)
    smooth_buffers[idx].append((x, y))
    xs = [p[0] for p in smooth_buffers[idx]]
    ys = [p[1] for p in smooth_buffers[idx]]
    return float(np.mean(xs)), float(np.mean(ys))


# ─────────────────────────────────────────────
# Gesture detectors
# ─────────────────────────────────────────────

def detect_face_landmarks(frame, face_landmarker):
    """Run FaceLandmarker on BGR frame. Returns results."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_landmarker.detect(mp_image)
    return results


def detect_vertical_motion(landmarks, w, h):
    global prev_face_center_y

    _, fh_y = get_smoothed_landmark(landmarks, LM_FOREHEAD, w, h)
    _, ch_y = get_smoothed_landmark(landmarks, LM_CHIN, w, h)

    center_y = (fh_y + ch_y) / 2

    if prev_face_center_y is None:
        prev_face_center_y = center_y
        return False

    movement = abs(center_y - prev_face_center_y)

    # dampen noise
    movement = movement * 0.7

    prev_face_center_y = center_y

    return movement > 8   # pixel threshold 5


def detect_eyes_open(landmarks, w, h):
    """
    Returns True if eyes are open.
    Uses eye height normalized by face height.
    """

    _, fh_y = get_smoothed_landmark(landmarks, LM_FOREHEAD, w, h)
    _, ch_y = get_smoothed_landmark(landmarks, LM_CHIN, w, h)
    face_height = abs(ch_y - fh_y) + 1e-6

    # Left eye
    _, le_top = get_smoothed_landmark(landmarks, 386, w, h)
    _, le_bot = get_smoothed_landmark(landmarks, 374, w, h)
    left_eye_height = abs(le_bot - le_top) / face_height

    # Right eye
    _, re_top = get_smoothed_landmark(landmarks, 159, w, h)
    _, re_bot = get_smoothed_landmark(landmarks, 145, w, h)
    right_eye_height = abs(re_bot - re_top) / face_height

    eye_open_ratio = (left_eye_height + right_eye_height) / 2

    return eye_open_ratio > 0.02   # threshold for "eyes open"





def detect_head_turn(landmarks, w, h):
    """
    Detect if head is turned left/right (yaw)
    """

    # Nose tip
    nx, _ = get_smoothed_landmark(landmarks, 1, w, h)

    # Face sides (ears)
    lx, _ = get_smoothed_landmark(landmarks, LM_LEFT_EAR, w, h)
    rx, _ = get_smoothed_landmark(landmarks, LM_RIGHT_EAR, w, h)

    face_width = abs(rx - lx) + 1e-6

    # normalize nose position
    nose_ratio = (nx - lx) / face_width

    # DEBUG
    # print(f"yaw ratio = {nose_ratio:.3f}")

    # center ≈ 0.5
    return nose_ratio < 0.35 or nose_ratio > 0.65



def detect_head_tilt(landmarks, w, h):
    """
    Detect vertical head tilt (looking up/down)
    Returns tilt value
    """

    _, fh_y = get_smoothed_landmark(landmarks, LM_FOREHEAD, w, h)
    _, ch_y = get_smoothed_landmark(landmarks, LM_CHIN, w, h)

    face_height = abs(ch_y - fh_y) + 1e-6

    # nose bridge approx (between eyes)
    _, nose_y = get_smoothed_landmark(landmarks, 6, w, h)

    # normalize
    tilt = (nose_y - fh_y) / face_height

    return tilt


def detect_eyes_looking_down(landmarks, w, h):
    """
    Detect if user is looking down (eyelids lowered)
    """

    # Left eye
    _, le_top = get_smoothed_landmark(landmarks, 386, w, h)
    _, le_bot = get_smoothed_landmark(landmarks, 374, w, h)

    # Right eye
    _, re_top = get_smoothed_landmark(landmarks, 159, w, h)
    _, re_bot = get_smoothed_landmark(landmarks, 145, w, h)

    left_height = abs(le_bot - le_top)
    right_height = abs(re_bot - re_top)

    avg_height = (left_height + right_height) / 2

    return avg_height < 6   # threshold (tune if needed)


def detect_head_tilt_lr(landmarks, w, h):

    # --- YAW BLOCK ---
    if detect_head_turn(landmarks, w, h):
        return None

    lx, ly = get_smoothed_landmark(landmarks, LM_LEYE_TOP, w, h)
    rx, ry = get_smoothed_landmark(landmarks, LM_REYE_TOP, w, h)

    dy = ry - ly
    dx = rx - lx + 1e-6

    slope = dy / dx

    if slope < -0.20:
        return 'left'
    elif slope > 0.20:
        return 'right'

    return None


def detect_eyebrows_raised(landmarks, w, h):

    # allow small motion, block only large movement
    movement_block = detect_vertical_motion(landmarks, w, h)

    if movement_block:
        return False, 0.0

    global brow_baseline, baseline_frames, is_calibrated

    _, fh_y = get_smoothed_landmark(landmarks, LM_FOREHEAD, w, h)
    _, ch_y = get_smoothed_landmark(landmarks, LM_CHIN, w, h)
    face_height = abs(ch_y - fh_y) + 1e-6

    lb_y = avg_y(landmarks, LM_LBROW, w, h)
    rb_y = avg_y(landmarks, LM_RBROW, w, h)

    _, le_top = get_smoothed_landmark(landmarks, 386, w, h)
    _, le_bot = get_smoothed_landmark(landmarks, 374, w, h)
    left_eye_center = (le_top + le_bot) / 2

    _, re_top = get_smoothed_landmark(landmarks, 159, w, h)
    _, re_bot = get_smoothed_landmark(landmarks, 145, w, h)
    right_eye_center = (re_top + re_bot) / 2

    left_delta = (left_eye_center - lb_y) / face_height
    right_delta = (right_eye_center - rb_y) / face_height

    delta = (left_delta + right_delta) / 2.0

    # --- Eye gate ---
    if not detect_eyes_open(landmarks, w, h):
        return False, 0.0
    
    # --- Eye direction gate (NEW) ---
    if detect_eyes_looking_down(landmarks, w, h):
        return False, 0.0

    # --- Head tilt gate ---
    tilt = detect_head_tilt(landmarks, w, h)
    # block if head is actively moving vertically

    # block if head is tilted sideways
    lx, ly = get_smoothed_landmark(landmarks, LM_LEYE_TOP, w, h)
    rx, ry = get_smoothed_landmark(landmarks, LM_REYE_TOP, w, h)

    slope = abs((ry - ly) / (rx - lx + 1e-6))

    if slope > 0.12:
        return False, 0.0


    if abs(tilt - 0.45) > 0.15:
        return False, 0.0

    # reject if looking too up or down
    if tilt < 0.25 or tilt > 0.65:
        return False, 0.0

    # --- CALIBRATION PHASE ---
    global is_calibrated

    if not is_calibrated:
        baseline_frames.append(delta)

        if len(baseline_frames) >= CALIBRATION_FRAMES:
            brow_baseline = sum(baseline_frames) / len(baseline_frames)
            is_calibrated = True
            print(f"[CALIBRATED] Brow baseline = {brow_baseline:.4f}")

        return False, 0.0   # don't detect during calibration

    # --- DETECTION PHASE ---
    raise_amount = delta - brow_baseline


    # --- DETECTION PHASE ---
    raise_amount = delta - brow_baseline

    if raise_amount > 0.010:  # threshold for raised brows
        if abs(left_delta - right_delta) < 0.04: # 0.03
            is_raised = True
        else:
            is_raised = False
    else:
        is_raised = False

    return is_raised, round(raise_amount, 4)


def detect_mouth_open(landmarks, w, h):
    """
    Mouth open (O shape) = vertical mouth opening exceeds threshold.
    This is DIFFERENT from talking — it's a deliberate wide-open mouth.
    Returns (is_open, open_ratio).
    """
    _,  mt_y = get_smoothed_landmark(landmarks, LM_MOUTH_TOP,    w, h)
    _,  mb_y = get_smoothed_landmark(landmarks, LM_MOUTH_BOTTOM, w, h)
    _,  fh_y = get_smoothed_landmark(landmarks, LM_FOREHEAD,     w, h)
    _,  ch_y = get_smoothed_landmark(landmarks, LM_CHIN,         w, h)

    face_height      = abs(ch_y - fh_y) + 1e-6
    mouth_open_ratio = abs(mb_y - mt_y) / face_height

    is_open = mouth_open_ratio >= MOUTH_OPEN_THRESHOLD
    return is_open, round(mouth_open_ratio, 4)


# ─────────────────────────────────────────────
# Action dispatcher — NO mouse movement at all
# Only sends keyboard shortcuts to focused window
# ─────────────────────────────────────────────
last_action_time = {}


def trigger_action(action):
    now = time.time()
    if now - last_action_time.get(action, 0.0) < COOLDOWN_SECONDS:
        return False
    last_action_time[action] = now

    if action == 'next':
        pyautogui.press('down')
        print("[ACTION] Next")

    elif action == 'previous':
        pyautogui.press('up')
        print("[ACTION] Previous")

    elif action == 'like':
        pyautogui.press('l')
        print("[ACTION] Like")

    return True


# ─────────────────────────────────────────────
# Gesture hold stability gate
# ─────────────────────────────────────────────
gesture_hold = collections.defaultdict(int)


def update_gesture_hold(detected_gesture):
    """Reset all other gestures, increment this one. Return True when ready."""
    for g in list(gesture_hold.keys()):
        if g != detected_gesture:
            gesture_hold[g] = 0
    gesture_hold[detected_gesture] += 1
    return gesture_hold[detected_gesture] >= REQUIRED_HOLD_FRAMES


# ─────────────────────────────────────────────
# Draw face mesh manually (tasks API landmarks)
# ─────────────────────────────────────────────
def draw_face_mesh(frame, landmarks, w, h):
    """Draw key landmark dots on the face — minimal and fast."""
    key_points = [
        LM_MOUTH_LEFT, LM_MOUTH_RIGHT, LM_MOUTH_TOP, LM_MOUTH_BOTTOM,
        LM_LEYE_TOP, LM_REYE_TOP,
        LM_FOREHEAD, LM_CHIN,
    ]

    # Add eyebrow points separately (since they are lists now)
    brow_points = LM_LBROW + LM_RBROW
    for idx in key_points:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1)

    # Draw mouth line
    ml = landmarks[LM_MOUTH_LEFT]
    mr = landmarks[LM_MOUTH_RIGHT]
    cv2.line(frame,
             (int(ml.x * w), int(ml.y * h)),
             (int(mr.x * w), int(mr.y * h)),
             (0, 200, 100), 1)

    # Draw eyebrow clusters (multiple points)
    for idx in LM_LBROW + LM_RBROW:
        lm = landmarks[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 2, (255, 180, 0), -1)

    # Visualize brow-to-eye distance (center line)
    lb_y = avg_y(landmarks, LM_LBROW, w, h)
    rb_y = avg_y(landmarks, LM_RBROW, w, h)

    lx, _ = get_smoothed_landmark(landmarks, 283, w, h)
    rx, _ = get_smoothed_landmark(landmarks, 53, w, h)

    _, le_y = get_smoothed_landmark(landmarks, LM_LEYE_TOP, w, h)
    _, re_y = get_smoothed_landmark(landmarks, LM_REYE_TOP, w, h)

    cv2.line(frame, (int(lx), int(lb_y)), (int(lx), int(le_y)), (255, 180, 0), 1)
    cv2.line(frame, (int(rx), int(rb_y)), (int(rx), int(re_y)), (255, 180, 0), 1)



# GESTURE_LABELS = {
#     'pause'   : ('HEAD NOD -> PLAY/PAUSE', (200, 200, 80)),
#     'next'    : ('BROWS UP -> NEXT VIDEO', (80, 180, 255)),
#     'previous': ('MOUTH OPEN -> PREV VIDEO', (255, 140, 60)),
#     'none'    : ('No gesture — waiting...', (140, 140, 140)),
# }
GESTURE_LABELS = {
    'like'    : ('MOUTH OPEN -> LIKE', (200, 200, 80)),
    'next'    : ('LEFT TILT -> NEXT', (80, 180, 255)),
    'previous': ('RIGHT TILT -> PREV', (255, 140, 60)),
    'none'    : ('No gesture — waiting...', (140, 140, 140)),
}


def draw_hud(frame, gesture, smile_r, brow_delta, mouth_r, fps):
    h, w = frame.shape[:2]

    # Semi-transparent top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 125), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Current gesture label
    label, color = GESTURE_LABELS.get(gesture, ('...', (200, 200, 200)))
    cv2.putText(frame, label, (12, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

    # Metric values    f"Smile:{smile_r:.3f}  "
    metrics = (
               f"Brow:{brow_delta:+.3f}  "
               f"Mouth:{mouth_r:.3f}  "
               f"FPS:{fps:.0f}")
    cv2.putText(frame, metrics, (12, 65),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (180, 180, 180), 1, cv2.LINE_AA)

    # Hold progress bar
    hold_count = gesture_hold.get(gesture if gesture != 'none' else '', 0)
    progress   = min(1.0, hold_count / REQUIRED_HOLD_FRAMES)
    bar_w      = int((w - 24) * progress)
    cv2.rectangle(frame, (12, 80),  (w - 12, 96), (40, 40, 40), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (12, 80), (12 + bar_w, 96), color, -1)
    cv2.putText(frame, "HOLD", (14, 93),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)

    # Instructions at bottom
    cv2.putText(frame, "Y=YouTube  I=Instagram  Q=Quit",
                (12, h - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (100, 100, 100), 1, cv2.LINE_AA)

    # Focus reminder — important!
    cv2.putText(frame, "Click Chrome first!",
                (w - 175, h - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 180, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def main():
    global current_mode

    pyautogui.FAILSAFE = False   # disable corner-abort so gestures near edges work

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam — check camera permissions.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)   # smaller = faster
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # Load FaceLandmarker via tasks API
    base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    fps_counter = collections.deque(maxlen=20)
    prev_time   = time.time()

    # Position window top-left so it stays out of Chrome's way
    cv2.namedWindow("Gesture Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Controller", 480, 360)
    cv2.moveWindow("Gesture Controller", 10, 10)

    print("=" * 55)
    print("  Reels & Shorts Gesture Controller")
    print("=" * 55)
    print("  GESTURES:")
    print("    Head Nod        ->  Play/Pause")
    print("    Eyebrows UP (hold)  ->  Next video")
    print("    Mouth OPEN (hold)   ->  Previous video")
    print()
    print("  KEYS (in webcam window):")
    print("    Y  ->  Switch to YouTube Shorts mode")
    print("    I  ->  Switch to Instagram Reels mode")
    print("    Q  ->  Quit")
    print()
    print("  IMPORTANT: Click on Chrome FIRST to give it focus!")
    print("  Current mode: YouTube Shorts")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # FPS
        now = time.time()
        fps_counter.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = float(np.mean(fps_counter))

        # Detect landmarks
        results = detect_face_landmarks(frame, face_landmarker)

        gesture    = 'none'
        smile_r    = 0.0
        brow_delta = 0.0
        mouth_r    = 0.0

        if results.face_landmarks:
            face_lm = results.face_landmarks[0]

            # Draw lightweight mesh
            draw_face_mesh(frame, face_lm, w, h)

            # Classify gesture
            # is_smiling,   smile_r    = detect_smile(face_lm, w, h)
            # is_brow_up,   brow_delta = detect_eyebrows_raised(face_lm, w, h)
            is_mouth_open, mouth_r   = detect_mouth_open(face_lm, w, h)
            # is_tilt = detect_head_tilt_lr(face_lm, w, h)
            tilt_dir = detect_head_tilt_lr(face_lm, w, h)


            # --- TILT STABILITY ---
            if tilt_dir in ['left', 'right']:
                tilt_frames += 1
            else:
                tilt_frames = 0

            # --- PRIORITY LOGIC ---
            if tilt_frames >= 3:
                gesture = 'next' if tilt_dir == 'left' else 'previous'

            elif is_mouth_open:
                gesture = 'like'

            else:
                gesture = 'none'
            # is_nod = detect_head_nod(face_lm, w, h)

            # Priority: tilt > mouth
            if tilt_dir == 'left':
                gesture = 'next'
            elif tilt_dir == 'right':
                gesture = 'previous'
            elif is_mouth_open:
                gesture = 'like'
            else:
                gesture = 'none'


            # --- ACTION LOGIC ---

            # Instant gestures (NO HOLD)
            if gesture in ['next', 'previous']:
                trigger_action(gesture)
                
                # reset hold so it doesn't interfere
                for g in list(gesture_hold.keys()):
                    gesture_hold[g] = 0


            # Hold-based gesture (LIKE)
            elif gesture == 'like':
                if update_gesture_hold(gesture):
                    fired = trigger_action(gesture)
                    if fired:
                        gesture_hold[gesture] = 0


            # No gesture
            else:
                for g in list(gesture_hold.keys()):
                    gesture_hold[g] = 0

        # Draw HUD
        draw_hud(frame, gesture, smile_r, brow_delta, mouth_r, fps)
        cv2.imshow("Gesture Controller", frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit.")
            break

    cap.release()
    face_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()