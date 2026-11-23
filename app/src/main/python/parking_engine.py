import io
import json
import time
import numpy as np
from PIL import Image

tracked_cars = {}
next_car_id = 1
entry_count = 0
exit_count = 0
counted_ids = set()
last_event_time = 0.0

def reset_counters():
    global entry_count, exit_count, tracked_cars, counted_ids, last_event_time
    entry_count = 0
    exit_count = 0
    tracked_cars.clear()
    counted_ids.clear()
    last_event_time = 0.0

def preprocess_frame(jpeg_bytes):
    pil_img = Image.open(io.BytesIO(bytes(jpeg_bytes))).convert("RGB")
    w, h = pil_img.size
    img_resized = pil_img.resize((320, 320))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_np.transpose(2, 0, 1), axis=0)
    return img_input.tobytes(), w, h

def nms_boxes(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    b = np.array(boxes, dtype=np.float32)
    x1, y1, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x2, y2 = x1 + w, y1 + h
    areas = w * h
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def postprocess_detections(raw_output_bytes, orig_w, orig_h, line_ratio=0.55):
    global next_car_id, entry_count, exit_count, tracked_cars, counted_ids, last_event_time

    outputs = np.frombuffer(bytes(raw_output_bytes), dtype=np.float32).reshape(1, 84, -1)
    predictions = np.squeeze(outputs).T

    line_y = int(orig_h * line_ratio)
    scale_x = orig_w / 320.0
    scale_y = orig_h / 320.0

    boxes = []
    scores = []
    for pred in predictions:
        car_conf = pred[6]  # Sadece otomobil (car)
        if car_conf > 0.40:
            cx, cy, bw, bh = pred[:4]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            boxes.append([x1, y1, int(bw * scale_x), int(bh * scale_y)])
            scores.append(float(car_conf))

    indices = nms_boxes(boxes, scores, 0.45)
    current_centers = []
    for i in indices:
        x, y, bw, bh = boxes[i]
        current_centers.append((x + bw // 2, y + bh // 2, x, y, bw, bh))

    event_data = None
    updated_cars = {}

    for cx, cy, x, y, bw, bh in current_centers:
        matched_id = None
        for cid, (prev_cx, prev_cy) in tracked_cars.items():
            if abs(cx - prev_cx) < 150 and abs(cy - prev_cy) < 150:
                matched_id = cid
                break

        if matched_id is None:
            matched_id = next_car_id
            next_car_id += 1
        else:
            prev_cx, prev_cy = tracked_cars[matched_id]
            if matched_id not in counted_ids:
                direction = None
                if prev_cy < line_y <= cy:
                    direction = "ENTRY"
                elif prev_cy > line_y >= cy:
                    direction = "EXIT"

                if direction is not None:
                    now = time.time()
                    # 3 saniyelik çizgi kilidi (dur-kalk / titreme koruması)
                    if now - last_event_time < 3.0:
                        updated_cars[matched_id] = (cx, cy)
                        continue
                    last_event_time = now
                    counted_ids.add(matched_id)

                    if direction == "ENTRY":
                        entry_count += 1
                    else:
                        exit_count += 1

                    event_data = {
                        "direction": direction,
                        "crop": [
                            max(0, x + int(bw * 0.12)),
                            max(0, y + int(bh * 0.50)),
                            min(orig_w, x + int(bw * 0.88)),
                            min(orig_h, y + bh)
                        ],
                        "time": time.strftime("%H:%M:%S")
                    }

        updated_cars[matched_id] = (cx, cy)

    tracked_cars = updated_cars

    return json.dumps({
        "entry": entry_count,
        "exit": exit_count,
        "event": event_data
    })