#!/usr/bin/env python3
import os
import time
import argparse
from collections import deque
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.cluster import KMeans
POSSIBLE_MODEL_PATHS = [
    r"C:\Users\ARYAN\runs\detect\train5\weights\best.pt",
]
def find_model_path():
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.isfile(path):
            return path
    return None
MODEL_PATH = find_model_path()
CSV_LOG = "intersection_final_log.csv"
CONF_THRES = 0.20
NMS_IOU_THRES = 0.65
VEHICLE_CLASS_IDS = {0, 1, 2, 3, 4, 5, 6, 7}
MIN_DETECTIONS_CALIB = 20
CALIB_SECONDS_DEFAULT = 5
CALIB_MAX_FRAMES = 300
BEV_W = 800
BEV_H = 600
TRACK_MAX_DISAPPEAR = 30
TRACK_ASSOC_MAX_DIST = 120
COLOR_GOING = (40, 180, 99)
COLOR_COMING = (36, 86, 255)
HUD_TEXT_COLOR = (255, 255, 255)
HUD_BG_COLOR = (0, 0, 0)
COUNT_LINE_COLOR = (255, 255, 255)
SHOW_GUIDES = False
HAS_DEEPSORT = False
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
    print("[INFO] deep-sort-realtime detected.")
except Exception:
    HAS_DEEPSORT = False
    print("[WARN] deep-sort-realtime not available.")
HAS_BYTETRACK = False
BYTETracker = None
STrack = None
try:
    from ultralytics.trackers.byte_tracker import BYTETracker as UBYTETracker, STrack as USTrack
    BYTETracker = UBYTETracker
    STrack = USTrack
    HAS_BYTETRACK = True
    print("[INFO] Ultralytics ByteTrack detected.")
except Exception:
    try:
        from yolox.tracker.byte_tracker import BYTETracker as YBYTETracker
        from yolox.tracker.byte_tracker import STrack as YSTrack
        BYTETracker = YBYTETracker
        STrack = YSTrack
        HAS_BYTETRACK = True
        print("[INFO] YOLOX ByteTrack detected.")
    except Exception:
        print("[WARN] ByteTrack not available.")
def project_points(M, points):
    if M is None or len(points) == 0:
        return np.array(points, dtype=np.float32)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    return projected
def poly_from_edge_fits(left_fit, right_fit, H):
    y_top = 0
    y_bot = H - 1
    lf = np.array(left_fit, dtype=float)
    rf = np.array(right_fit, dtype=float)
    left_top_x = lf[0] * y_top + lf[1]
    left_bot_x = lf[0] * y_bot + lf[1]
    right_top_x = rf[0] * y_top + rf[1]
    right_bot_x = rf[0] * y_bot + rf[1]
    src = np.array(
        [[left_top_x, y_top], [right_top_x, y_top], [right_bot_x, y_bot], [left_bot_x, y_bot]],
        dtype=np.float32,
    )
    return src
def fit_edge_lines_from_centroids(centroids, H, slices=8):
    if len(centroids) < 10:
        return (0.0, 0.15 * H), (0.0, 0.85 * H)
    pts = np.array(centroids, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    x_lo, x_hi = np.percentile(xs, [10, 90])
    y_lo, y_hi = np.percentile(ys, [10, 90])
    mask = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 10:
        return (0.0, 0.15 * H), (0.0, 0.85 * H)
    Hmin, Hmax = int(np.min(ys)), int(np.max(ys))
    if Hmax - Hmin < 20:
        Hmin, Hmax = 0, H - 1
    slice_h = max(1, (Hmax - Hmin) // slices)
    left_pts, right_pts = [], []
    for i in range(slices):
        y0 = Hmin + i * slice_h
        y1 = min(Hmax, y0 + slice_h)
        mask_slice = (ys >= y0) & (ys <= y1)
        if mask_slice.sum() < 2:
            continue
        xs_slice = xs[mask_slice]
        ys_slice = ys[mask_slice]
        left_pts.append((float(np.mean(ys_slice)), float(np.min(xs_slice))))
        right_pts.append((float(np.mean(ys_slice)), float(np.max(xs_slice))))
    if len(left_pts) < 2 or len(right_pts) < 2:
        return (0.0, 0.15 * H), (0.0, 0.85 * H)
    ly = np.array([p[0] for p in left_pts])
    lx = np.array([p[1] for p in left_pts])
    ry = np.array([p[0] for p in right_pts])
    rx = np.array([p[1] for p in right_pts])
    try:
        la, lb = np.polyfit(ly, lx, 1)
        ra, rb = np.polyfit(ry, rx, 1)
        if la > 5 or ra > 5:
            return (0.0, 0.15 * H), (0.0, 0.85 * H)
        return (la, lb), (ra, rb)
    except Exception:
        return (0.0, 0.15 * H), (0.0, 0.85 * H)
def compute_perspective_transform_from_centroids(centroids, W, H):
    if len(centroids) < 8:
        return None, None, None
    left_fit, right_fit = fit_edge_lines_from_centroids(centroids, H, slices=8)
    src = poly_from_edge_fits(left_fit, right_fit, H)
    dst = np.array(
        [[0, 0], [BEV_W - 1, 0], [BEV_W - 1, BEV_H - 1], [0, BEV_H - 1]], dtype=np.float32
    )
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv, src
    except Exception:
        return None, None, src
class SimpleTracker:
    def __init__(self, max_disappear=TRACK_MAX_DISAPPEAR, max_dist=TRACK_ASSOC_MAX_DIST):
        self.next_id = 1
        self.tracks = {}
        self.max_disappear = max_disappear
        self.max_dist = max_dist
    def _add(self, centroid):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            "centroid": tuple(map(float, centroid)),
            "history": deque([tuple(map(float, centroid))], maxlen=32),
            "disappear": 0,
        }
        return tid
    def update(self, detections):
        if len(self.tracks) == 0:
            for c in detections:
                self._add(c)
            return {tid: t["centroid"] for tid, t in self.tracks.items()}
        ids = list(self.tracks.keys())
        prev = np.array([self.tracks[i]["centroid"] for i in ids], dtype=np.float32)
        dets = (
            np.array(detections, dtype=np.float32)
            if len(detections)
            else np.zeros((0, 2), dtype=np.float32)
        )
        if dets.shape[0] == 0:
            for tid in ids:
                self.tracks[tid]["disappear"] += 1
            for tid in list(self.tracks.keys()):
                if self.tracks[tid]["disappear"] > self.max_disappear:
                    del self.tracks[tid]
            return {tid: t["centroid"] for tid, t in self.tracks.items()}
        D = np.linalg.norm(prev[:, None, :] - dets[None, :, :], axis=2)
        matched_prev = set()
        matched_det = set()
        while True:
            idx = np.unravel_index(np.argmin(D, axis=None), D.shape)
            i, j = idx
            val = D[i, j]
            if not np.isfinite(val) or val > self.max_dist:
                break
            tid = ids[i]
            det = tuple(map(float, dets[j]))
            self.tracks[tid]["centroid"] = det
            self.tracks[tid]["history"].append(det)
            self.tracks[tid]["disappear"] = 0
            matched_prev.add(i)
            matched_det.add(j)
            D[i, :] = np.inf
            D[:, j] = np.inf
            if (D == np.inf).all():
                break
        for j in range(dets.shape[0]):
            if j in matched_det:
                continue
            self._add(tuple(map(float, dets[j])))
        for i_idx, tid in enumerate(ids):
            if i_idx in matched_prev:
                continue
            self.tracks[tid]["disappear"] += 1
            if self.tracks[tid]["disappear"] > self.max_disappear:
                del self.tracks[tid]
        return {tid: self.tracks[tid]["centroid"] for tid in self.tracks}
def safe_parse_box(box):
    try:
        if hasattr(box, "xyxy"):
            if hasattr(box.xyxy, "cpu"):
                coords = box.xyxy.cpu().numpy().flatten()
            elif hasattr(box.xyxy, "numpy"):
                coords = box.xyxy.numpy().flatten()
            else:
                coords = np.array(box.xyxy).flatten()
            x1, y1, x2, y2 = map(int, coords[:4])
        else:
            coords = np.array(box).flatten()
            x1, y1, x2, y2 = map(int, coords[:4])
    except Exception as e:
        print(f"[WARN] Box parsing failed: {e}")
        return 0, 0, 0, 0, -1, 0.0
    try:
        if hasattr(box, "cls"):
            if hasattr(box.cls, "cpu"):
                cls = int(box.cls.cpu().numpy().flatten()[0])
            elif hasattr(box.cls, "numpy"):
                cls = int(box.cls.numpy().flatten()[0])
            else:
                cls = int(np.array(box.cls).flatten()[0])
        else:
            cls = -1
    except Exception as e:
        print(f"[WARN] Class parsing failed: {e}")
        cls = -1
    try:
        if hasattr(box, "conf"):
            if hasattr(box.conf, "cpu"):
                conf = float(box.conf.cpu().numpy().flatten()[0])
            elif hasattr(box.conf, "numpy"):
                conf = float(box.conf.numpy().flatten()[0])
            else:
                conf = float(np.array(box.conf).flatten()[0])
        else:
            conf = 0.0
    except Exception as e:
        print(f"[WARN] Confidence parsing failed: {e}")
        conf = 0.0
    return x1, y1, x2, y2, cls, conf
def filter_overlapping_boxes(det_boxes, det_scores, det_classes, iou_threshold=NMS_IOU_THRES):
    if len(det_boxes) == 0:
        return det_boxes, det_scores, det_classes
    boxes = np.array(det_boxes, dtype=float)
    scores = np.array(det_scores, dtype=float)
    order = scores.argsort()[::-1]
    keep_idxs = []
    while order.size > 0:
        i = order[0]
        keep_idxs.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter + 1e-9
        iou = inter / union
        keep_mask = iou <= iou_threshold
        order = rest[keep_mask]
    keep_idxs = sorted(keep_idxs)
    return [det_boxes[i] for i in keep_idxs], [det_scores[i] for i in keep_idxs], [det_classes[i] for i in keep_idxs]
class ByteTrackWrapper:
    def __init__(self, track_thresh=0.25, match_thresh=0.8, frame_rate=30):
        self.impl = "ultralytics"
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.bt = BYTETracker(
            track_thresh=self.track_thresh,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate
        )
    def update(self, det_boxes, det_scores, img_size):
        results = []
        for box, score in zip(det_boxes, det_scores):
            results.append({
                "bbox": box,
                "score": score,
                "class": 0
            })
        tracks = self.bt.update(results, img_size, img_size)
        out = []
        for t in tracks:
            tlbr = t.tlbr
            track_id = t.track_id
            out.append((track_id, tlbr))
        return out
def run(source, calib_seconds=CALIB_SECONDS_DEFAULT, tracker_name="auto", debug=False, conf_override=None, nms_override=None, show_guides=None):
    global SHOW_GUIDES
    if show_guides is not None:
        SHOW_GUIDES = bool(show_guides)
    print("=== intersection_auto_control.py (ROBUST) ===")
    if MODEL_PATH is None:
        print("[ERROR] No model found! Ensure best.pt exists in one of:")
        for p in POSSIBLE_MODEL_PATHS:
            print("  -", p)
        return
    print("MODEL_PATH:", MODEL_PATH)
    if conf_override is not None:
        global CONF_THRES
        CONF_THRES = float(conf_override)
    if nms_override is not None:
        global NMS_IOU_THRES
        NMS_IOU_THRES = float(nms_override)
    try:
        model = YOLO(MODEL_PATH)
        print("[INFO] Model loaded.")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(dummy_img, verbose=False, conf=0.1)
        print("[INFO] Model sanity test OK.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    try:
        src = int(source)
    except Exception:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Cannot open source:", source)
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[INFO] Source opened: {W}x{H} @ {FPS:.1f}fps")
    tracker_type = tracker_name.lower()
    if tracker_type == "auto":
        if HAS_DEEPSORT:
            tracker_type = "deepsort"
        elif HAS_BYTETRACK:
            tracker_type = "bytetrack"
        else:
            tracker_type = "simple"
    print(f"[INFO] Tracker selected: {tracker_type}")
    deepsort = None
    bytetrack = None
    simple_tracker = None
    if tracker_type == "deepsort" and HAS_DEEPSORT:
        try:
            deepsort = DeepSort(max_age=TRACK_MAX_DISAPPEAR)
            print("[INFO] DeepSORT initialized.")
        except Exception as e:
            print("[WARN] DeepSORT init failed:", e)
            deepsort = None
    if tracker_type == "bytetrack" and HAS_BYTETRACK:
        try:
            bytetrack = ByteTrackWrapper(
                fps=FPS, track_thresh=max(0.1, CONF_THRES * 0.8), match_thresh=0.8, track_buffer=TRACK_MAX_DISAPPEAR
            )
            print("[INFO] ByteTrack initialized.")
        except Exception as e:
            print("[WARN] ByteTrack init failed:", e)
            bytetrack = None
    if deepsort is None and bytetrack is None:
        simple_tracker = SimpleTracker(max_disappear=TRACK_MAX_DISAPPEAR, max_dist=TRACK_ASSOC_MAX_DIST)
        print("[INFO] Using fallback SimpleTracker.")
    if not os.path.exists(CSV_LOG):
        pd.DataFrame(
            columns=["time", "frame", "going_active", "coming_active", "going_total", "coming_total", "total_active"]
        ).to_csv(CSV_LOG, index=False)
    print(f"[INFO] Calibration for ~{calib_seconds}s. Please ensure moving traffic.")
    calib_start = time.time()
    frame_count = 0
    all_centroids = []
    while time.time() - calib_start < calib_seconds and frame_count < CALIB_MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            results = model(frame, verbose=False, conf=CONF_THRES)
            detection_count = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2, cls, conf = safe_parse_box(box)
                    if cls in VEHICLE_CLASS_IDS and conf >= CONF_THRES:
                        if (x2 - x1) * (y2 - y1) > 100:
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            all_centroids.append((cx, cy))
                            detection_count += 1
        except Exception as e:
            print(f"[WARN] Detection failed in calibration: {e}")
        disp = frame.copy()
        remaining = max(0, int(calib_seconds - (time.time() - calib_start)))
        cv2.putText(
            disp,
            f"Calibrating... {remaining}s | Dets: {len(all_centroids)}",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 180, 255),
            2,
        )
        cv2.imshow("Calibrating (press q to skip)", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Calibration skipped by user.")
            break
        frame_count += 1
    print(f"[INFO] Calibration collected {len(all_centroids)} points.")
    if len(all_centroids) < MIN_DETECTIONS_CALIB:
        print("[WARN] Few calibration detections; BEV may be imprecise.")
    M, Minv, src_trap = compute_perspective_transform_from_centroids(all_centroids, W, H)
    if M is not None:
        print("[INFO] BEV homography computed.")
    else:
        print("[WARN] No BEV homography; running in image space.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    going_total = 0
    coming_total = 0
    track_store = {}
    paused = False
    frame_idx = 0
    last_time = time.time()
    print("[INFO] Main loop. Keys: q=quit, p=pause, s=snapshot, r=reset, d=debug")
    min_cross_travel = 6.0 if M is not None else 10.0
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("[INFO] Source ended.")
                break
            now_t = time.time()
            det_boxes, det_scores, det_classes = [], [], []
            try:
                results = model(frame, verbose=False, conf=CONF_THRES)
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2, cls, conf = safe_parse_box(box)
                        if cls in VEHICLE_CLASS_IDS and conf >= CONF_THRES:
                            if (x2 - x1) * (y2 - y1) > 120:
                                det_boxes.append((x1, y1, x2, y2))
                                det_scores.append(conf)
                                det_classes.append(cls)
            except Exception as e:
                print(f"[ERROR] Detection failed on frame {frame_idx}: {e}")
            det_boxes, det_scores, det_classes = filter_overlapping_boxes(det_boxes, det_scores, det_classes, NMS_IOU_THRES)
            active_tracks = {}
            if deepsort is not None:
                ds_dets = []
                for box, score, cls in zip(det_boxes, det_scores, det_classes):
                    x1, y1, x2, y2 = box
                    ds_dets.append(([x1, y1, x2 - x1, y2 - y1], float(score), int(cls)))
                try:
                    tracks = deepsort.update_tracks(ds_dets, frame=frame)
                except Exception:
                    tracks = deepsort.update_tracks(ds_dets)
                for tr in tracks:
                    try:
                        if not tr.is_confirmed():
                            continue
                        tid = int(tr.track_id)
                        tlbr = tr.to_ltrb()
                        if tlbr is None:
                            continue
                        x1, y1, x2, y2 = map(int, tlbr)
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        active_tracks[tid] = {"centroid": (cx, cy), "bbox": (x1, y1, x2, y2)}
                    except Exception:
                        continue
            elif bytetrack is not None:
                bt_out = bytetrack.update(det_boxes, det_scores, (H, W))
                for t in bt_out:
                    tid = t["id"]
                    x1, y1, x2, y2 = t["tlbr"]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    active_tracks[tid] = {"centroid": (cx, cy), "bbox": (x1, y1, x2, y2)}
            if simple_tracker is not None:
                det_centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in det_boxes]
                map_res = simple_tracker.update(det_centroids)
                for tid, cent in map_res.items():
                    active_tracks[tid] = {"centroid": (int(cent[0]), int(cent[1])), "bbox": None}
            bev_centroids = {}
            for tid, info in active_tracks.items():
                cx, cy = info["centroid"]
                if M is not None:
                    p = project_points(M, [(cx, cy)])[0]
                    bev_centroids[tid] = (float(p[0]), float(p[1]))
                else:
                    bev_centroids[tid] = (float(cx), float(cy))
            lane_centers = []
            if len(bev_centroids) >= 2:
                xs = np.array([p[0] for p in bev_centroids.values()]).reshape(-1, 1)
                try:
                    n_clusters = min(2, len(bev_centroids))
                    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(xs)
                    lane_centers = sorted(list(km.cluster_centers_.flatten()))
                except Exception:
                    lane_centers = []
            if len(lane_centers) == 0:
                lane_centers = [BEV_W * 0.33, BEV_W * 0.66] if M is not None else [W * 0.4, W * 0.6]
            center_line_y = BEV_H / 2 if M is not None else H / 2
            LOCKOUT_FRAMES = 12
            MIN_TRACK_AGE = 5
            min_cross_travel = 6.0 if M is not None else 14.0
            going_active = 0
            coming_active = 0
            for tid, (cx_bev, cy_bev) in bev_centroids.items():
                if tid not in track_store:
                    track_store[tid] = {
                        "history": deque(maxlen=16),
                        "age": 0,
                        "y_ema": None,
                        "prev_y_ema": None,
                        "prev_side": None,
                        "last_cross_frame": -10_000,
                        "counted_dir": None
                    }
                ts = track_store[tid]
                ts["age"] += 1
                ts["history"].append((cx_bev, cy_bev))
                if ts["y_ema"] is None:
                    ts["y_ema"] = float(cy_bev)
                    ts["prev_y_ema"] = float(cy_bev)
                else:
                    ts["prev_y_ema"] = ts["y_ema"]
                    ts["y_ema"] = 0.7 * ts["y_ema"] + 0.3 * float(cy_bev)
                dy_inst = ts["y_ema"] - ts["prev_y_ema"]
                if dy_inst < 0:
                    going_active += 1
                else:
                    coming_active += 1
                margin = 1.0
                if ts["y_ema"] < center_line_y - margin:
                    side = -1
                elif ts["y_ema"] > center_line_y + margin:
                    side = +1
                else:
                    side = 0
                if side == 0:
                    side = ts["prev_side"] if ts["prev_side"] is not None else 0
                can_count = (
                    ts["prev_side"] is not None and
                    side != ts["prev_side"] and
                    (frame_idx - ts["last_cross_frame"] > LOCKOUT_FRAMES) and
                    ts["age"] >= MIN_TRACK_AGE and
                    abs(ts["y_ema"] - ts["prev_y_ema"]) + abs(cy_bev - ts["history"][0][1]) >= min_cross_travel
                )
                if can_count and ts["counted_dir"] is None:
                    if ts["prev_side"] == -1 and side == +1:
                        coming_total += 1
                        ts["counted_dir"] = "coming"
                        ts["last_cross_frame"] = frame_idx
                        if debug: print(f"[COUNT] ID {tid} COMING -> total {coming_total}")
                    elif ts["prev_side"] == +1 and side == -1:
                        going_total += 1
                        ts["counted_dir"] = "going"
                        ts["last_cross_frame"] = frame_idx
                        if debug: print(f"[COUNT] ID {tid} GOING -> total {going_total}")
                ts["prev_side"] = side
            total_active = going_active + coming_active
            fps = 1.0 / (now_t - last_time + 1e-9)
            last_time = now_t
            hud_text1 = f"Active -> Going: {going_active}  Coming: {coming_active}  Total: {total_active}"
            hud_text2 = f"Cumulative -> Going: {going_total}  Coming: {coming_total}  FPS: {fps:.1f}"
            hud_text3 = f"Detections: {len(det_boxes)}  Tracks: {len(active_tracks)}    Tracker: {tracker_type}"
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, 8), (560, 106), HUD_BG_COLOR, -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, hud_text1, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_TEXT_COLOR, 2)
            cv2.putText(frame, hud_text2, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_TEXT_COLOR, 1)
            cv2.putText(frame, hud_text3, (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if M is not None and Minv is not None:
                img_center_pt = project_points(Minv, [(BEV_W * 0.5, BEV_H * 0.5)])[0]
                count_line_y = int(img_center_pt[1])
            else:
                count_line_y = int(H / 2)
            cv2.line(frame, (0, count_line_y), (W, count_line_y), COUNT_LINE_COLOR, 3)
            cv2.putText(
                frame,
                "COUNT LINE",
                (W - 140, max(20, count_line_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COUNT_LINE_COLOR,
                2,
            )
            for tid, info in active_tracks.items():
                cx, cy = info["centroid"]
                bbox = info.get("bbox", None)
                ts = track_store.get(tid, {})
                if ts.get("counted_dir") == "going":
                    color = COLOR_GOING; status = "GOING"; thick = 3
                elif ts.get("counted_dir") == "coming":
                    color = COLOR_COMING; status = "COMING"; thick = 3
                else:
                    color = (150, 150, 150); status = "TRACK"; thick = 2
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                    label = f"ID{tid}"
                    if status != "TRACK":
                        label += f":{status}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0] + 8, y1 - 2), color, -1)
                    cv2.putText(frame, label, (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.circle(frame, (cx, cy), 3, color, -1)
                else:
                    cv2.circle(frame, (int(cx), int(cy)), 8, color, -1)
                    cv2.circle(frame, (int(cx), int(cy)), 12, color, 2)
                    cv2.putText(frame, f"ID{tid}", (int(cx) + 15, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                hist = ts.get("history", deque())
                if len(hist) > 3 and M is not None and Minv is not None and status in ("GOING", "COMING"):
                    bev_pts = list(hist)[-6:]
                    img_pts = project_points(Minv, bev_pts)
                    for i in range(1, len(img_pts)):
                        p1 = (int(img_pts[i - 1][0]), int(img_pts[i - 1][1]))
                        p2 = (int(img_pts[i][0]), int(img_pts[i][1]))
                        cv2.line(frame, p1, p2, color, 1)
            if SHOW_GUIDES and src_trap is not None:
                try:
                    pts = src_trap.reshape(-1, 2).astype(np.int32)
                    cv2.polylines(frame, [pts], True, (100, 100, 200), 1)
                except Exception:
                    pass
            if SHOW_GUIDES:
                for idx, lc in enumerate(lane_centers):
                    if M is not None and Minv is not None:
                        bev_pt = (lc, BEV_H * 0.1)
                        orig_pt = project_points(Minv, [bev_pt])[0]
                        px, py = int(orig_pt[0]), max(20, int(orig_pt[1]))
                    else:
                        px = int(W * (0.25 + idx * 0.5))
                        py = 30
                    color = (0, 150, 255) if idx % 2 == 0 else (255, 150, 0)
                    cv2.circle(frame, (px, py), 6, color, -1)
                    cv2.circle(frame, (px, py), 8, color, 1)
            cv2.putText(frame, f"Conf:{CONF_THRES:.2f}  NMS:{NMS_IOU_THRES:.2f}", (W - 240, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Intersection Auto Control (q=quit,p=pause,s=snap)", frame)
            try:
                now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row = {
                    "time": now_ts,
                    "frame": frame_idx,
                    "going_active": int(going_active),
                    "coming_active": int(coming_active),
                    "going_total": int(going_total),
                    "coming_total": int(coming_total),
                    "total_active": int(total_active),
                }
                pd.DataFrame([row]).to_csv(CSV_LOG, mode="a", header=False, index=False)
            except Exception as e:
                if debug:
                    print(f"[WARN] CSV logging failed: {e}")
            frame_idx += 1
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("p"):
            paused = not paused
            print("[INFO] paused" if paused else "[INFO] resumed")
        elif k == ord("s") and not paused:
            os.makedirs("snapshots", exist_ok=True)
            path = os.path.join("snapshots", f"snap_{int(time.time())}.jpg")
            cv2.imwrite(path, frame)
            print("[INFO] snapshot saved:", path)
        elif k == ord("r"):
            going_total = 0
            coming_total = 0
            track_store.clear()
            print("[INFO] Counters reset")
        elif k == ord("d"):
            print(
                f"[DEBUG] Active tracks: {len(active_tracks)}  Dets: {len(det_boxes)}  BEV: {len(bev_centroids)}  Conf:{CONF_THRES} NMS:{NMS_IOU_THRES}"
            )
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] finished. CSV:", CSV_LOG)
    print(f"[FINAL] Total vehicles -> Going: {going_total}, Coming: {coming_total}")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robust Traffic Management (YOLOv8 + DeepSORT/ByteTrack)")
    ap.add_argument("--source", required=True, help="video path or webcam index, e.g., 0")
    ap.add_argument("--calib", type=int, default=CALIB_SECONDS_DEFAULT, help="calibration seconds (default 6)")
    ap.add_argument("--conf", type=float, default=None, help="override detection confidence threshold")
    ap.add_argument("--nms", type=float, default=None, help="override NMS IoU threshold")
    ap.add_argument(
        "--tracker",
        type=str,
        default="auto",
        choices=["auto", "deepsort", "bytetrack", "simple"],
        help="tracker to use (default: auto)",
    )
    ap.add_argument("--show-guides", action="store_true", help="show ROI & lane guide overlays (no red direction line)")
    ap.add_argument("--debug", action="store_true", help="verbose debug logs")
    args = ap.parse_args()
    run(
        args.source,
        calib_seconds=args.calib,
        tracker_name=args.tracker,
        debug=args.debug,
        conf_override=args.conf,
        nms_override=args.nms,
        show_guides=args.show_guides,
    )