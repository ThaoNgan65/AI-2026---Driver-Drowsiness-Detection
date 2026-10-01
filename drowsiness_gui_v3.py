"""
Driver Drowsiness Detection - App v3
------------------------------------
Logic detection giữ nguyên + độ tin cậy:
  1) Watchdog / Heartbeat
  2) Camera Obstructed
  3) Error logging (file + console)
"""

import cv2
import time
import csv
import os
import logging
import threading
import traceback
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import winsound
from datetime import datetime

MODEL_PATH = "face_landmarker.task"
LOG_PATH = "alerts_log.csv"
ERROR_LOG_PATH = "system_errors.log"

# ====================== NGƯỠNG ======================
EAR_THRESHOLD_NORMAL = 0.18
EAR_THRESHOLD_STRICT = 0.13
EAR_THRESHOLD_YAW = 0.25
EAR_THRESHOLD_EXTREME = 0.28
EAR_VERY_LOW = 0.12
EAR_THRESHOLD_LOWLIGHT = 0.23

EYE_CLOSED_TIME_LIMIT = 1.8
MAR_THRESHOLD = 0.65
YAWN_TIME_LIMIT = 0.9
BEEP_INTERVAL = 0.2
FACE_LOST_GRACE = 0.7

PITCH_HIGH = 0.40
YAW_HIGH = 0.35
YAW_EXTREME = 0.65
SMILE_RATIO_THRESHOLD = 0.6

NOD_DROP_THRESHOLD = 0.45
NOD_RECOVER_THRESHOLD = 0.41
NOD_COUNT_NEEDED = 2
NOD_TIME_WINDOW = 6.5
NOD_HOLD_TIME = 1.5

DEEP_PITCH_THRESHOLD = 0.55
DEEP_HOLD_TIME = 2.2
DEEP_NO_FACE_GRACE = 3.5

LOW_LIGHT_MEAN = 55
GLARE_CLIP_RATIO = 0.12
GLARE_EYE_MEAN_MIN = 215.0
# Rung/chuyển động: KHÔNG khóa nod — chỉ siết thời gian giữ cúi/ngẩng
MOTION_DIFF_THRESH = 14.0
NOD_PITCH_SMOOTH = 0.35
NOD_MIN_GAP = 0.50
NOD_DWELL_NORMAL = 0.10     # phải giữ cúi/ngẩng tối thiểu (s)
NOD_DWELL_NOISY = 0.25      # khi frame nhiễu (rung): giữ lâu hơn mới đếm 1 gật

EAR_SMOOTH_ALPHA = 0.45
OPEN_RESET_FRAMES = 4

SUNGLASS_RATIO_NORMAL = 0.62
SUNGLASS_RATIO_LOWLIGHT = 0.48
SUNGLASS_CONFIRM_FRAMES = 5

CAM_FAIL_REOPEN = 30

# ----- An toàn / watchdog -----
HEARTBEAT_TIMEOUT = 2.5          # giây không có frame/detect → SYSTEM ERROR
WATCHDOG_CHECK_MS = 500
MAX_DETECT_RESTARTS = 2
OBSTRUCTED_HOLD_SEC = 2.0
# 1) Flatness trên ảnh blur (bắt che mọi màu)
# 2) Structure recovery bằng CLAHE (phòng tối có mặt → lộ cạnh → không obstructed)
OBSTRUCTED_LAP_VAR_MAX = 18.0
OBSTRUCTED_LOW_VAR_RATIO = 0.70
OBSTRUCTED_LOCAL_STD_MAX = 7.0
OBSTRUCTED_CENTER_STD_MAX = 8.0
OBSTRUCTED_CENTER_LAP_MAX = 14.0
OBSTRUCTED_LOCAL_KERNEL = 15
# Sau CLAHE: nếu còn cấu trúc → cảnh thật (kể cả thiếu sáng)
OBSTRUCTED_RECOVER_LAP_MIN = 28.0
OBSTRUCTED_RECOVER_EDGE_MIN = 0.015
# ====================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]
SKIN_REF = [10, 50, 280]

# ----- Error logging -----
def setup_error_logger():
    logger = logging.getLogger("drowsiness")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = setup_error_logger()


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    v1 = euclidean_distance(pts[1], pts[5])
    v2 = euclidean_distance(pts[2], pts[4])
    hori = euclidean_distance(pts[0], pts[3])
    return (v1 + v2) / (2.0 * hori + 1e-6)


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]
    return euclidean_distance(pts[2], pts[3]) / (euclidean_distance(pts[0], pts[1]) + 1e-6)


def compute_head_pitch_score(landmarks, w, h):
    nose, chin = landmarks[1], landmarks[152]
    le, re = landmarks[33], landmarks[263]
    eye_mid_y = ((le.y + re.y) / 2.0) * h
    denom = chin.y * h - eye_mid_y
    if abs(denom) < 1e-3:
        return 0.0
    return (nose.y * h - eye_mid_y) / denom


def compute_head_yaw_score(landmarks, w, h):
    nose, le, re = landmarks[1], landmarks[33], landmarks[263]
    eye_mid_x = ((le.x + re.x) / 2.0) * w
    eye_dist = abs((re.x - le.x) * w) + 1e-6
    return (nose.x * w - eye_mid_x) / (eye_dist / 2.0)


def center_roi_stats(gray):
    h, w = gray.shape[:2]
    y1, y2 = int(h * 0.18), int(h * 0.82)
    x1, x2 = int(w * 0.22), int(w * 0.78)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0
    return float(np.mean(roi >= 250)), float(np.mean(roi))


def enhance_lighting(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    clip_ratio, _ = center_roi_stats(gray)

    if mean_val < LOW_LIGHT_MEAN:
        gamma = 1.35
        table = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
        ).astype("uint8")
        bright = cv2.LUT(bgr, table)
        lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        return out, "low_light"

    if clip_ratio > GLARE_CLIP_RATIO:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lf = l.astype(np.float32)
        hi = lf > 180
        lf[hi] = 180.0 + (lf[hi] - 180.0) * 0.40
        if clip_ratio > 0.20:
            lf *= 0.92
        l2 = np.clip(lf, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        return out, "glare"

    return bgr, "normal"


def roi_gray_mean(bgr, landmarks, indices, w, h, pad=3):
    xs = [int(landmarks[i].x * w) for i in indices]
    ys = [int(landmarks[i].y * h) for i in indices]
    if not xs:
        return None
    x1, x2 = max(0, min(xs) - pad), min(w - 1, max(xs) + pad)
    y1, y2 = max(0, min(ys) - pad), min(h - 1, max(ys) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    g = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(np.mean(g))


def is_sunglasses_relative(bgr, landmarks, w, h, ratio_max):
    left_m = roi_gray_mean(bgr, landmarks, LEFT_EYE, w, h)
    right_m = roi_gray_mean(bgr, landmarks, RIGHT_EYE, w, h)
    skin_m = roi_gray_mean(bgr, landmarks, SKIN_REF, w, h, pad=6)
    if left_m is None or right_m is None or skin_m is None or skin_m < 1.0:
        return False
    eye_m = (left_m + right_m) / 2.0
    return (eye_m / skin_m) < ratio_max


def eyes_overexposed(bgr, landmarks, w, h):
    m1 = roi_gray_mean(bgr, landmarks, LEFT_EYE, w, h, pad=4)
    m2 = roi_gray_mean(bgr, landmarks, RIGHT_EYE, w, h, pad=4)
    if m1 is None or m2 is None:
        return False
    return ((m1 + m2) * 0.5) >= GLARE_EYE_MEAN_MIN

def is_color_obstructed(bgr):
    """
    Phát hiện che bằng vật liệu để ánh sáng xuyên qua (ngón tay, vải mỏng...)
    tạo ra màu đơn sắc áp đảo toàn khung hình — điển hình là cam/đỏ.
    Không cần quan tâm texture vì bản chất là một mảng màu gần như đồng nhất.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    h, w = h_ch.shape[:2]

    # Vùng trung tâm, tránh viền/góc bị vignette
    cy1, cy2 = int(h * 0.15), int(h * 0.85)
    cx1, cx2 = int(w * 0.15), int(w * 0.85)
    h_roi = h_ch[cy1:cy2, cx1:cx2].astype(np.float32)
    s_roi = s_ch[cy1:cy2, cx1:cx2].astype(np.float32)
    v_roi = v_ch[cy1:cy2, cx1:cx2].astype(np.float32)

    # OpenCV hue: 0-180. Cam-đỏ nằm khoảng 0-20 hoặc 160-180
    warm_mask = (h_roi <= 20) | (h_roi >= 160)
    warm_ratio = float(np.mean(warm_mask))

    mean_sat = float(np.mean(s_roi))
    mean_val = float(np.mean(v_roi))

    # Độ đồng nhất của hue trên TOÀN vùng trung tâm (không chỉ trong warm_mask)
    # — vật che ánh sáng xuyên qua cho hue rất đồng đều, cảnh thật (da người,
    # đồ vật màu cam trong nền) hiếm khi phủ kín + đồng đều như vậy
    hue_std = float(np.std(h_roi))

    is_warm_dominant = warm_ratio > 0.6
    is_saturated = mean_sat > 40
    is_uniform = hue_std < 25
    is_not_too_dark_or_bright = 10 < mean_val < 250

    return is_warm_dominant and is_saturated and is_uniform and is_not_too_dark_or_bright


def is_frame_obstructed(bgr):
    """
    Camera bị che gần hết — mọi màu.
    Hai bước:
      A) Flatness sau blur mạnh (bắt tay/vải/giấy, không cần màu đen).
      B) Structure recovery bằng CLAHE — phòng tối có mặt/cảnh sẽ lộ cạnh
         → KHÔNG obstructed; che ống kính thì CLAHE vẫn gần như phẳng.
    """
    # Check màu trước — rẻ và đặc trưng cho case ngón tay/vật che xuyên sáng
    if is_color_obstructed(bgr):
        return True

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return False

    # ----- B) Structure recovery (ưu tiên: thiếu sáng ≠ che) -----
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enh_blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    recover_lap = float(cv2.Laplacian(enh_blur, cv2.CV_64F).var())
    recover_edge = float(np.mean(cv2.Canny(enhanced, 40, 120) > 0))
    if (
        recover_lap >= OBSTRUCTED_RECOVER_LAP_MIN
        or recover_edge >= OBSTRUCTED_RECOVER_EDGE_MIN
    ):
        return False

    # ----- A) Flatness trên raw (che mọi màu) -----
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    lap_var = float(cv2.Laplacian(blur, cv2.CV_64F).var())

    k = OBSTRUCTED_LOCAL_KERNEL
    if k % 2 == 0:
        k += 1
    f = blur.astype(np.float32)
    mean = cv2.blur(f, (k, k))
    mean_sq = cv2.blur(f * f, (k, k))
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    low_var_ratio = float(np.mean(local_std < OBSTRUCTED_LOCAL_STD_MAX))

    cy1, cy2 = int(h * 0.20), int(h * 0.80)
    cx1, cx2 = int(w * 0.20), int(w * 0.80)
    center_std_map = local_std[cy1:cy2, cx1:cx2]
    center_std = float(np.mean(center_std_map)) if center_std_map.size else 999.0
    center = blur[cy1:cy2, cx1:cx2]
    center_lap = (
        float(cv2.Laplacian(center, cv2.CV_64F).var()) if center.size else lap_var
    )

    # Phải phẳng trên raw VÀ không phục hồi cấu trúc sau CLAHE
    if (
        low_var_ratio >= OBSTRUCTED_LOW_VAR_RATIO
        and lap_var < OBSTRUCTED_LAP_VAR_MAX
        and center_std < OBSTRUCTED_CENTER_STD_MAX
    ):
        return True

    if (
        center_lap < OBSTRUCTED_CENTER_LAP_MAX
        and center_std < OBSTRUCTED_CENTER_STD_MAX
        and low_var_ratio >= 0.60
        and lap_var < OBSTRUCTED_LAP_VAR_MAX * 1.35
    ):
        return True

    return False


def ensure_log_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["start_time", "end_time", "duration_sec", "alert_types"]
            )


def log_alert(start_ts, end_ts, reasons):
    try:
        ensure_log_header()
        dur = max(0.0, end_ts - start_ts)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S"),
                f"{dur:.1f}",
                " + ".join(reasons) if reasons else "",
            ])
    except Exception as e:
        log.error("log_alert failed: %s", e)


class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection v3")
        self.root.geometry("980x680")
        self.root.configure(bg="#121212")
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.running = False
        self.thread = None
        self.cap = None
        self.landmarker = None
        self.photo = None

        self.ear = self.mar = self.pitch = self.yaw = 0.0
        self.nod_count = 0
        self.closed_dur = self.yawn_dur = self.deep_dur = 0.0
        self.fps = 0
        self.status = "STOPPED"
        self.reason = ""
        self.eye_progress = self.yawn_progress = 0.0
        self.env_flag = ""
        self.eye_unreliable = False
        self.current_th_display = EAR_THRESHOLD_NORMAL

        # Watchdog / system health
        self.last_heartbeat = 0.0
        self.system_error = False
        self.system_error_msg = ""
        self.detect_restarts = 0
        self.camera_obstructed = False
        self._obstructed_since = None

        self.intro_frame = tk.Frame(root, bg="#121212")
        self.main_frame = tk.Frame(root, bg="#1a1a1a")
        self.build_intro()
        self.build_main()
        self.show_intro()
        ensure_log_header()
        log.info("App started")

    def build_intro(self):
        f = self.intro_frame
        tk.Label(
            f, text="DRIVER DROWSINESS DETECTION",
            font=("Segoe UI", 22, "bold"), bg="#121212", fg="#00e5a0"
        ).pack(pady=(48, 6))
        tk.Label(
            f, text="Hệ thống phát hiện dấu hiệu buồn ngủ của tài xế  |  v3",
            font=("Segoe UI", 11), bg="#121212", fg="#cccccc"
        ).pack(pady=(0, 20))

        info = (
            "Chức năng:\n"
            "  • Mắt nhắm (EAR)  • Ngáp (MAR)  • Gật gù  • Cúi sâu giữ lâu\n"
            "  • Thích ứng: kính râm, thiếu sáng, glare, rung camera\n"
            "  • Watchdog, camera bị che, ghi log lỗi hệ thống\n"
            "  • Ghi log cảnh báo (alerts_log.csv)\n\n"
            "Lưu ý: Chỉ phát hiện dấu hiệu quan sát được từ khuôn mặt,\n"
            "không khẳng định trạng thái buồn ngủ bên trong."
        )
        tk.Label(
            f, text=info, font=("Segoe UI", 10), bg="#121212", fg="#aaaaaa",
            justify="left"
        ).pack(padx=70, pady=8)

        tk.Button(
            f, text="Bắt đầu / Vào ứng dụng",
            font=("Segoe UI", 13, "bold"), bg="#00c853", fg="white",
            width=24, height=2, cursor="hand2", command=self.show_main
        ).pack(pady=32)

        tk.Label(
            f, text="Sáng tạo trẻ Quốc gia 2026  |  Bảng C – AI",
            font=("Segoe UI", 9), bg="#121212", fg="#666666"
        ).pack(side=tk.BOTTOM, pady=16)

    def show_intro(self):
        self.main_frame.pack_forget()
        self.intro_frame.pack(fill=tk.BOTH, expand=True)

    def build_main(self):
        f = self.main_frame
        top = tk.Frame(f, bg="#222222", height=48)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Button(
            top, text="← Giới thiệu", font=("Segoe UI", 9),
            bg="#333", fg="#ddd", relief=tk.FLAT, command=self.back_to_intro
        ).pack(side=tk.LEFT, padx=10, pady=10)

        self.status_label = tk.Label(
            top, text="STOPPED", font=("Segoe UI", 14, "bold"),
            bg="#222222", fg="#888"
        )
        self.status_label.pack(side=tk.LEFT, padx=14)

        self.reason_label = tk.Label(
            top, text="", font=("Segoe UI", 10), bg="#222222", fg="#ff5252"
        )
        self.reason_label.pack(side=tk.LEFT, padx=6)

        self.env_label = tk.Label(
            top, text="", font=("Segoe UI", 9), bg="#222222", fg="#ffab40"
        )
        self.env_label.pack(side=tk.RIGHT, padx=12)

        body = tk.Frame(f, bg="#1a1a1a")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        cam_frame = tk.Frame(body, bg="#000")
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(cam_frame, bg="#000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        panel = tk.Frame(body, bg="#252525", width=270)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        panel.pack_propagate(False)

        tk.Label(panel, text="Thông số", font=("Segoe UI", 11, "bold"),
                 bg="#252525", fg="#eee").pack(pady=(10, 4))

        self.vars = {}
        for name, key in [
            ("EAR", "ear"), ("MAR", "mar"), ("Pitch", "pitch"),
            ("Yaw", "yaw"), ("Nod", "nod"), ("Deep", "deep"),
            ("Closed", "closed"), ("Yawn", "yawn"), ("FPS", "fps"),
        ]:
            row = tk.Frame(panel, bg="#252525")
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=name, font=("Segoe UI", 9),
                     bg="#252525", fg="#aaa", width=8, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            self.vars[key] = var
            tk.Label(row, textvariable=var, font=("Consolas", 10, "bold"),
                     bg="#252525", fg="#00e5a0", anchor="e").pack(side=tk.RIGHT)

        tk.Label(panel, text="Eye", font=("Segoe UI", 8),
                 bg="#252525", fg="#888").pack(anchor="w", padx=12, pady=(8, 2))
        self.eye_bar = ttk.Progressbar(panel, length=230, mode="determinate", maximum=100)
        self.eye_bar.pack(padx=12)

        tk.Label(panel, text="Yawn", font=("Segoe UI", 8),
                 bg="#252525", fg="#888").pack(anchor="w", padx=12, pady=(6, 2))
        self.yawn_bar = ttk.Progressbar(panel, length=230, mode="determinate", maximum=100)
        self.yawn_bar.pack(padx=12)

        btn_box = tk.Frame(panel, bg="#252525")
        btn_box.pack(side=tk.BOTTOM, pady=12)

        self.btn_start = tk.Button(
            btn_box, text="Start Camera", width=12, font=("Segoe UI", 10, "bold"),
            bg="#00c853", fg="white", command=self.start
        )
        self.btn_start.pack(pady=3)
        self.btn_stop = tk.Button(
            btn_box, text="Stop Camera", width=12, font=("Segoe UI", 10, "bold"),
            bg="#ffab00", fg="black", command=self.stop, state=tk.DISABLED
        )
        self.btn_stop.pack(pady=3)
        tk.Button(
            btn_box, text="Exit", width=12, font=("Segoe UI", 10, "bold"),
            bg="#e53935", fg="white", command=self.on_exit
        ).pack(pady=3)

    def show_main(self):
        self.intro_frame.pack_forget()
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def back_to_intro(self):
        self.stop()
        self.main_frame.pack_forget()
        self.intro_frame.pack(fill=tk.BOTH, expand=True)

    def init_landmarker(self):
        try:
            base = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            return True
        except Exception as e:
            log.error("MediaPipe init failed: %s\n%s", e, traceback.format_exc())
            self.landmarker = None
            return False

    def close_landmarker(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception as e:
                log.warning("close_landmarker: %s", e)
            self.landmarker = None

    def open_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                log.warning("cap.release: %s", e)
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok = self.cap.isOpened()
            if not ok:
                log.error("Camera open failed (isOpened=False)")
            return ok
        except Exception as e:
            log.error("open_camera: %s\n%s", e, traceback.format_exc())
            self.cap = None
            return False

    def start(self):
        if self.running:
            return
        self.system_error = False
        self.system_error_msg = ""
        self.detect_restarts = 0
        self.camera_obstructed = False
        self._obstructed_since = None
        self.close_landmarker()
        if not self.init_landmarker():
            self.system_error = True
            self.system_error_msg = "SYSTEM ERROR - DETECTION STOPPED"
            self.status = "SYSTEM ERROR"
            log.error("Start aborted: landmarker init failed")
            return
        if not self.open_camera():
            self.system_error = True
            self.system_error_msg = "SYSTEM ERROR - DETECTION STOPPED"
            self.status = "SYSTEM ERROR"
            log.error("Start aborted: camera open failed")
            return
        self.running = True
        self.last_heartbeat = time.time()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status = "NORMAL"
        self.reason = ""
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
        self.update_gui()
        self._schedule_watchdog()
        log.info("Detection started")

    def stop(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if not self.system_error:
            self.status = "STOPPED"
            self.reason = ""
        self.env_flag = ""
        self.camera_obstructed = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                log.warning("stop cap.release: %s", e)
            self.cap = None
        self.close_landmarker()
        log.info("Detection stopped by user")

    def on_exit(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                log.warning("exit cap.release: %s", e)
        self.close_landmarker()
        log.info("App exit")
        self.root.destroy()

    def _schedule_watchdog(self):
        """Heartbeat: detection im lặng quá lâu → SYSTEM ERROR (+ restart có giới hạn)."""
        if not self.running:
            return
        try:
            if (
                not self.system_error
                and self.last_heartbeat > 0
                and (time.time() - self.last_heartbeat) > HEARTBEAT_TIMEOUT
            ):
                log.error(
                    "Watchdog: no heartbeat for %.1fs",
                    time.time() - self.last_heartbeat,
                )
                if self.detect_restarts < MAX_DETECT_RESTARTS:
                    self.detect_restarts += 1
                    log.warning(
                        "Attempting detection restart %d/%d",
                        self.detect_restarts,
                        MAX_DETECT_RESTARTS,
                    )
                    self._try_restart_detection()
                else:
                    self.system_error = True
                    self.system_error_msg = "SYSTEM ERROR - DETECTION STOPPED"
                    self.status = "SYSTEM ERROR"
                    self.reason = ""
                    self.running = False
                    try:
                        self.btn_start.config(state=tk.NORMAL)
                        self.btn_stop.config(state=tk.DISABLED)
                    except Exception:
                        pass
                    log.error("Detection stopped permanently after max restarts")
        except Exception as e:
            log.error("watchdog: %s\n%s", e, traceback.format_exc())

        if self.running and not self.system_error:
            self.root.after(WATCHDOG_CHECK_MS, self._schedule_watchdog)

    def _try_restart_detection(self):
        """Restart nhẹ: đóng/mở lại camera + landmarker, start thread mới."""
        try:
            self.running = False
            time.sleep(0.15)
            self.close_landmarker()
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    log.warning("restart release: %s", e)
                self.cap = None
            if not self.init_landmarker() or not self.open_camera():
                self.system_error = True
                self.system_error_msg = "SYSTEM ERROR - DETECTION STOPPED"
                self.status = "SYSTEM ERROR"
                log.error("Restart failed: init/camera")
                return
            self.running = True
            self.last_heartbeat = time.time()
            self.system_error = False
            self.status = "NORMAL"
            self.thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.thread.start()
            log.info("Detection restarted OK")
        except Exception as e:
            self.system_error = True
            self.system_error_msg = "SYSTEM ERROR - DETECTION STOPPED"
            self.status = "SYSTEM ERROR"
            self.running = False
            log.error("Restart exception: %s\n%s", e, traceback.format_exc())

    def detection_loop(self):
        if self.cap is None or self.landmarker is None:
            self.running = False
            log.error("detection_loop: cap or landmarker is None")
            return

        start_time = time.time()
        prev_time = time.time()
        prev_gray = None
        fail_reads = 0

        eye_closed_start = None
        yawn_start = None
        face_lost_start = None
        is_alarm_on = False
        last_beep_time = 0.0
        last_closed = last_yawn = 0.0
        is_dropped = False
        nod_times = []
        nodding_hold_until = 0.0
        pitch_nod = 0.0
        pitch_nod_ready = False
        last_nod_count_time = 0.0
        drop_since = None
        recover_since = None
        deep_start = None
        deep_duration = 0.0
        was_in_deep_pose = False
        sunglass_streak = 0
        ear_smooth = 0.0
        ear_smooth_ready = False
        open_streak = 0

        alert_active = False
        alert_start_ts = 0.0
        alert_reasons_snap = []

        try:
            while self.running and not self.system_error:
                ret, frame = self.cap.read()
                if not ret:
                    fail_reads += 1
                    if fail_reads >= CAM_FAIL_REOPEN:
                        log.warning("Camera read fail x%d — reconnect", fail_reads)
                        if self.open_camera():
                            fail_reads = 0
                            self.last_heartbeat = time.time()
                        else:
                            time.sleep(0.5)
                    else:
                        time.sleep(0.02)
                    continue
                fail_reads = 0
                self.last_heartbeat = time.time()

                if not self.running or self.landmarker is None:
                    break

                frame = cv2.flip(frame, 1)

                # ----- Camera obstructed (che kín) -----
                if is_frame_obstructed(frame):
                    if self._obstructed_since is None:
                        self._obstructed_since = time.time()
                    elif time.time() - self._obstructed_since >= OBSTRUCTED_HOLD_SEC:
                        self.camera_obstructed = True
                else:
                    self._obstructed_since = None
                    self.camera_obstructed = False

                if self.camera_obstructed:
                    # Không báo buồn ngủ; reset timer detection
                    eye_closed_start = yawn_start = None
                    last_closed = last_yawn = 0.0
                    is_dropped = False
                    drop_since = recover_since = None
                    nod_times = []
                    nodding_hold_until = 0.0
                    deep_start = None
                    deep_duration = 0.0
                    was_in_deep_pose = False
                    open_streak = 0
                    ear_smooth_ready = False
                    if alert_active:
                        log_alert(alert_start_ts, time.time(), alert_reasons_snap)
                        alert_active = False
                    is_alarm_on = False
                    self.status = "CAMERA OBSTRUCTED"
                    self.reason = ""
                    self.env_flag = "obstructed"
                    self.ear = self.mar = self.pitch = self.yaw = 0.0
                    self.nod_count = 0
                    self.closed_dur = self.yawn_dur = self.deep_dur = 0.0
                    self.eye_progress = self.yawn_progress = 0.0
                    self.eye_unreliable = False
                    cv2.putText(
                        frame, "CAMERA OBSTRUCTED", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
                    )
                    rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_show).resize(
                        (640, 480), Image.Resampling.BILINEAR
                    )
                    self.photo = ImageTk.PhotoImage(image=img)
                    now = time.time()
                    self.fps = int(1.0 / (now - prev_time + 1e-6))
                    prev_time = now
                    time.sleep(0.01)
                    continue

                frame, light_flag = enhance_lighting(frame)
                h, w = frame.shape[:2]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                high_motion = False
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    if float(np.mean(cv2.absdiff(gray, prev_gray))) > MOTION_DIFF_THRESH:
                        high_motion = True
                prev_gray = gray

                env_parts = []
                if light_flag == "low_light":
                    env_parts.append("low_light")
                elif light_flag == "glare":
                    env_parts.append("glare")
                if high_motion:
                    env_parts.append("motion")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = int((time.time() - start_time) * 1000)

                try:
                    result = self.landmarker.detect_for_video(mp_image, ts)
                except ValueError as e:
                    log.warning("detect_for_video ValueError: %s", e)
                    break
                except Exception as e:
                    log.error("detect_for_video: %s\n%s", e, traceback.format_exc())
                    break

                ear_value = mar_value = pitch = yaw = 0.0
                closed_duration = last_closed
                yawn_duration = last_yawn
                nod_count = 0
                is_nodding_alarm = False
                is_deep_alarm = False
                current_th = EAR_THRESHOLD_NORMAL
                eye_unreliable = False

                if result.face_landmarks:
                    face_lost_start = None
                    fl = result.face_landmarks[0]

                    left_ear = eye_aspect_ratio(fl, LEFT_EYE, w, h)
                    right_ear = eye_aspect_ratio(fl, RIGHT_EYE, w, h)
                    mar_value = mouth_aspect_ratio(fl, MOUTH, w, h)
                    pitch = compute_head_pitch_score(fl, w, h)
                    yaw = compute_head_yaw_score(fl, w, h)

                    ratio_th = (
                        SUNGLASS_RATIO_LOWLIGHT
                        if light_flag == "low_light"
                        else SUNGLASS_RATIO_NORMAL
                    )
                    if is_sunglasses_relative(frame, fl, w, h, ratio_th):
                        sunglass_streak += 1
                    else:
                        sunglass_streak = 0
                    eye_unreliable = sunglass_streak >= SUNGLASS_CONFIRM_FRAMES
                    if eye_unreliable:
                        env_parts.append("sunglass")

                    if eyes_overexposed(frame, fl, w, h):
                        eye_unreliable = True
                        if "glare" not in env_parts:
                            env_parts.append("glare")

                    mw = euclidean_distance(
                        (fl[61].x * w, fl[61].y * h),
                        (fl[291].x * w, fl[291].y * h),
                    )
                    fw = euclidean_distance(
                        (fl[33].x * w, fl[33].y * h),
                        (fl[263].x * w, fl[263].y * h),
                    )
                    is_smiling = (mw / (fw + 1e-6)) > SMILE_RATIO_THRESHOLD

                    if yaw > YAW_EXTREME:
                        ear_value = left_ear
                        current_th = EAR_THRESHOLD_EXTREME
                    elif yaw < -YAW_EXTREME:
                        ear_value = right_ear
                        current_th = EAR_THRESHOLD_EXTREME
                    else:
                        ear_value = (left_ear + right_ear) / 2.0
                        if abs(yaw) > YAW_HIGH:
                            current_th = EAR_THRESHOLD_YAW
                        elif pitch > PITCH_HIGH or is_smiling:
                            current_th = EAR_THRESHOLD_STRICT
                        else:
                            current_th = EAR_THRESHOLD_NORMAL

                    if ear_value < EAR_VERY_LOW:
                        current_th = EAR_VERY_LOW

                    if light_flag == "low_light" and not eye_unreliable:
                        if current_th < EAR_THRESHOLD_LOWLIGHT:
                            current_th = EAR_THRESHOLD_LOWLIGHT

                    # EAR mọi frame (kể cả blur)
                    if eye_unreliable:
                        if eye_closed_start is not None:
                            closed_duration = time.time() - eye_closed_start
                        else:
                            closed_duration = 0.0
                    else:
                        if not ear_smooth_ready:
                            ear_smooth = ear_value
                            ear_smooth_ready = True
                        else:
                            ear_smooth = (
                                EAR_SMOOTH_ALPHA * ear_value
                                + (1.0 - EAR_SMOOTH_ALPHA) * ear_smooth
                            )
                        if ear_smooth < current_th:
                            open_streak = 0
                            if eye_closed_start is None:
                                eye_closed_start = time.time()
                            closed_duration = time.time() - eye_closed_start
                        else:
                            open_streak += 1
                            if open_streak >= OPEN_RESET_FRAMES:
                                eye_closed_start = None
                                closed_duration = 0.0
                                open_streak = 0
                            elif eye_closed_start is not None:
                                closed_duration = time.time() - eye_closed_start
                            else:
                                closed_duration = 0.0

                    last_closed = closed_duration

                    # Yawn + Deep: không khóa khi rung
                    if mar_value > MAR_THRESHOLD:
                        if yawn_start is None:
                            yawn_start = time.time()
                        yawn_duration = time.time() - yawn_start
                    else:
                        yawn_start = None
                        yawn_duration = 0.0
                    last_yawn = yawn_duration

                    if pitch > DEEP_PITCH_THRESHOLD:
                        was_in_deep_pose = True
                        if deep_start is None:
                            deep_start = time.time()
                        deep_duration = time.time() - deep_start
                    else:
                        was_in_deep_pose = False
                        deep_start = None
                        deep_duration = 0.0
                    is_deep_alarm = deep_duration >= DEEP_HOLD_TIME

                    # Nod: luôn chạy. Nhiễu/rung → dwell dài hơn (vẫn nhận gật thật)
                    now = time.time()
                    if not pitch_nod_ready:
                        pitch_nod = pitch
                        pitch_nod_ready = True
                    else:
                        pitch_nod = (
                            NOD_PITCH_SMOOTH * pitch
                            + (1.0 - NOD_PITCH_SMOOTH) * pitch_nod
                        )
                    dwell = NOD_DWELL_NOISY if high_motion else NOD_DWELL_NORMAL
                    nod_times = [t for t in nod_times if now - t <= NOD_TIME_WINDOW]

                    if pitch_nod > NOD_DROP_THRESHOLD:
                        recover_since = None
                        if drop_since is None:
                            drop_since = now
                        if (now - drop_since) >= dwell:
                            is_dropped = True
                    elif pitch_nod < NOD_RECOVER_THRESHOLD:
                        drop_since = None
                        if is_dropped:
                            if recover_since is None:
                                recover_since = now
                            if (now - recover_since) >= dwell:
                                if now - last_nod_count_time >= NOD_MIN_GAP:
                                    nod_times.append(now)
                                    last_nod_count_time = now
                                    if len(nod_times) >= NOD_COUNT_NEEDED:
                                        nodding_hold_until = now + NOD_HOLD_TIME
                                        nod_times = []
                                is_dropped = False
                                recover_since = None
                        else:
                            recover_since = None
                    else:
                        recover_since = None

                    nod_count = len(nod_times)
                    is_nodding_alarm = now < nodding_hold_until

                    for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                        lm = fl[idx]
                        cv2.circle(
                            frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 220, 100), -1
                        )
                else:
                    sunglass_streak = 0
                    if face_lost_start is None:
                        face_lost_start = time.time()
                    lost = time.time() - face_lost_start

                    if was_in_deep_pose:
                        if deep_start is None:
                            deep_start = time.time() - deep_duration
                        deep_duration = time.time() - deep_start
                        is_deep_alarm = deep_duration >= DEEP_HOLD_TIME
                        if lost >= DEEP_NO_FACE_GRACE:
                            eye_closed_start = yawn_start = None
                            last_closed = last_yawn = closed_duration = yawn_duration = 0.0
                            is_dropped = False
                            drop_since = recover_since = None
                            nod_times = []
                            nodding_hold_until = 0.0
                            deep_start = None
                            deep_duration = 0.0
                            was_in_deep_pose = False
                            is_deep_alarm = False
                            is_alarm_on = False
                            open_streak = 0
                            ear_smooth_ready = False
                    else:
                        is_deep_alarm = False
                        if lost >= FACE_LOST_GRACE:
                            eye_closed_start = yawn_start = None
                            last_closed = last_yawn = closed_duration = yawn_duration = 0.0
                            is_dropped = False
                            drop_since = recover_since = None
                            nod_times = []
                            nodding_hold_until = 0.0
                            deep_start = None
                            deep_duration = 0.0
                            was_in_deep_pose = False
                            is_alarm_on = False
                            open_streak = 0
                            ear_smooth_ready = False

                eye_alarm = (not eye_unreliable) and (
                    closed_duration >= EYE_CLOSED_TIME_LIMIT
                )
                yawn_alarm = yawn_duration >= YAWN_TIME_LIMIT
                should_alarm = (
                    eye_alarm or yawn_alarm or is_nodding_alarm or is_deep_alarm
                )

                if should_alarm:
                    reasons = []
                    if eye_alarm:
                        reasons.append("Mắt nhắm")
                    if yawn_alarm:
                        reasons.append("Ngáp")
                    if is_nodding_alarm:
                        reasons.append("Gật gù")
                    if is_deep_alarm:
                        reasons.append("Cúi sâu")

                    if not is_alarm_on:
                        is_alarm_on = True
                        alert_active = True
                        alert_start_ts = time.time()
                        alert_reasons_snap = list(reasons)
                        try:
                            winsound.Beep(1000, 160)
                            last_beep_time = time.time()
                        except Exception as e:
                            log.warning("Beep failed: %s", e)
                    else:
                        alert_reasons_snap = list(reasons)
                        if time.time() - last_beep_time >= BEEP_INTERVAL:
                            try:
                                winsound.Beep(1000, 120)
                                last_beep_time = time.time()
                            except Exception as e:
                                log.warning("Beep failed: %s", e)
                    self.reason = " + ".join(reasons)
                    self.status = "WARNING"
                else:
                    if alert_active:
                        log_alert(alert_start_ts, time.time(), alert_reasons_snap)
                        alert_active = False
                    is_alarm_on = False
                    self.reason = ""
                    self.status = "NORMAL"

                self.ear, self.mar = ear_value, mar_value
                self.pitch, self.yaw = pitch, yaw
                self.nod_count = nod_count
                self.closed_dur, self.yawn_dur = closed_duration, yawn_duration
                self.deep_dur = deep_duration
                self.eye_progress = min(
                    100.0, closed_duration / EYE_CLOSED_TIME_LIMIT * 100
                )
                self.yawn_progress = min(100.0, yawn_duration / YAWN_TIME_LIMIT * 100)
                self.eye_unreliable = eye_unreliable
                self.current_th_display = current_th
                self.env_flag = ", ".join(env_parts)

                now = time.time()
                self.fps = int(1.0 / (now - prev_time + 1e-6))
                prev_time = now

                if eye_unreliable and "sunglass" in self.env_flag:
                    cv2.putText(
                        frame, "EAR skipped (sunglass)", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                    )
                elif eye_unreliable:
                    cv2.putText(
                        frame, "EAR skipped (glare/eyes)", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                    )
                elif light_flag == "low_light":
                    cv2.putText(
                        frame, f"Low-light EAR th={current_th:.2f}", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                    )
                if high_motion:
                    cv2.putText(
                        frame, "Motion: stricter nod dwell", (12, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                    )

                rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_show).resize(
                    (640, 480), Image.Resampling.BILINEAR
                )
                self.photo = ImageTk.PhotoImage(image=img)
                time.sleep(0.006)

        except Exception as e:
            log.error("detection_loop crashed: %s\n%s", e, traceback.format_exc())
            self.system_error = True
            self.system_error_msg = "SYSTEM ERROR - DETECTION FAILED"
            self.status = "SYSTEM ERROR"
            self.running = False
            # Cập nhật UI ngay trên main thread
            try:
                self.root.after(0, self._show_system_error_ui)
            except Exception as ui_e:
                log.warning("Could not schedule SYSTEM ERROR UI: %s", ui_e)

        if alert_active:
            log_alert(alert_start_ts, time.time(), alert_reasons_snap)

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                log.warning("loop end release: %s", e)
            self.cap = None

    def _show_system_error_ui(self):
        """Gọi từ main thread khi detection_loop gặp exception."""
        try:
            self.status_label.config(text="SYSTEM ERROR", fg="#ff1744")
            self.reason_label.config(
                text=self.system_error_msg or "SYSTEM ERROR - DETECTION FAILED"
            )
            self.env_label.config(text="")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
        except Exception as e:
            log.warning("_show_system_error_ui: %s", e)

    def update_gui(self):
        # Ưu tiên trạng thái hệ thống
        if self.system_error:
            self.status_label.config(
                text="SYSTEM ERROR", fg="#ff1744"
            )
            self.reason_label.config(
                text=self.system_error_msg or "SYSTEM ERROR - DETECTION FAILED"
            )
            self.env_label.config(text="")
            try:
                self.btn_start.config(state=tk.NORMAL)
                self.btn_stop.config(state=tk.DISABLED)
            except Exception:
                pass
            if self.photo is not None:
                self.video_label.configure(image=self.photo)
                self.video_label.image = self.photo
            return

        if self.camera_obstructed and self.running:
            self.status_label.config(text="CAMERA OBSTRUCTED", fg="#ff9100")
            self.reason_label.config(text="Camera bị che / không có hình")
            self.env_label.config(text="obstructed")
            self.vars["ear"].set("—")
            self.vars["mar"].set("—")
            self.vars["pitch"].set("—")
            self.vars["yaw"].set("—")
            self.vars["nod"].set("0")
            self.vars["deep"].set("0.0s")
            self.vars["closed"].set("0.0s")
            self.vars["yawn"].set("0.0s")
            self.vars["fps"].set(str(self.fps))
            self.eye_bar["value"] = 0
            self.yawn_bar["value"] = 0
            if self.photo is not None:
                self.video_label.configure(image=self.photo)
                self.video_label.image = self.photo
            if self.running:
                self.root.after(30, self.update_gui)
            return

        if not self.running and self.status == "STOPPED":
            self.status_label.config(text="STOPPED", fg="#888")
            self.reason_label.config(text="")
            self.env_label.config(text="")
            return

        if self.status == "WARNING":
            self.status_label.config(text="WARNING", fg="#ff5252")
            self.reason_label.config(text=self.reason)
        else:
            self.status_label.config(text="NORMAL", fg="#00e676")
            self.reason_label.config(text="")

        self.env_label.config(text=self.env_flag)

        if self.eye_unreliable:
            ear_txt = f"{self.ear:.3f} (skip)"
        else:
            ear_txt = f"{self.ear:.3f} th={self.current_th_display:.2f}"
        self.vars["ear"].set(ear_txt)
        self.vars["mar"].set(f"{self.mar:.3f}")
        self.vars["pitch"].set(f"{self.pitch:.3f}")
        self.vars["yaw"].set(f"{self.yaw:.2f}")
        self.vars["nod"].set(str(self.nod_count))
        self.vars["deep"].set(f"{self.deep_dur:.1f}s")
        self.vars["closed"].set(f"{self.closed_dur:.1f}s")
        self.vars["yawn"].set(f"{self.yawn_dur:.1f}s")
        self.vars["fps"].set(str(self.fps))
        self.eye_bar["value"] = self.eye_progress
        self.yawn_bar["value"] = self.yawn_progress

        if self.photo is not None:
            self.video_label.configure(image=self.photo)
            self.video_label.image = self.photo

        if self.running:
            self.root.after(30, self.update_gui)


if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()