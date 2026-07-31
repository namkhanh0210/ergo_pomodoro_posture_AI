import base64
import gc
import io
import math
import os
import re
import cv2
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from pydantic import BaseModel

app = FastAPI(title="ErgoAI - Lightweight ONNX Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

desk_session = None
pose_session = None

def create_onnx_options():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return opts

def get_desk_session():
    global desk_session
    if desk_session is None:
        desk_session = ort.InferenceSession("yolov8n.onnx", sess_options=create_onnx_options())
    return desk_session

def get_pose_session():
    global pose_session
    if pose_session is None:
        pose_session = ort.InferenceSession("yolov8n-pose.onnx", sess_options=create_onnx_options())
    return pose_session

AUDIO_CACHE = {}
MAX_CACHE_SIZE = 5
SCREEN_REAL_WIDTH_CM = 35.0

COCO_CLASSES = {
    0: 'person', 39: 'bottle', 41: 'cup', 62: 'tvmonitor',
    63: 'laptop', 64: 'mouse', 66: 'keyboard'
}

class ImageData(BaseModel):
    image_base64: str
    is_calibration: bool
    baseline_eye_dist: float = 0.0
    baseline_shoulder_y: float = 0.0

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def cv2_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def clean_text_for_tts(text):
    cleaned = re.sub(r'[\#\*\_\`\>\-\[\]\(\)]', ' ', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def get_voice_base64(text: str, lang='en') -> str:
    if not text:
        return ""
    if len(AUDIO_CACHE) > MAX_CACHE_SIZE:
        AUDIO_CACHE.clear()
        
    cache_key = f"{lang}_{text}"
    if cache_key not in AUDIO_CACHE:
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            AUDIO_CACHE[cache_key] = base64.b64encode(fp.read()).decode('utf-8')
        except Exception:
            return ""
    return AUDIO_CACHE[cache_key]

def run_desk_onnx(img, conf_thresh=0.25, iou_thresh=0.45):
    h_orig, w_orig = img.shape[:2]
    img_resized = cv2.resize(img, (480, 480))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    session = get_desk_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})[0]

    out = outputs[0]
    if out.shape[0] < out.shape[1]:
        out = out.T

    boxes = out[:, :4]
    scores = out[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences >= conf_thresh
    boxes, class_ids, confidences = boxes[mask], class_ids[mask], confidences[mask]

    if len(boxes) == 0:
        return {}

    boxes_nms = []
    for cx, cy, w, h in boxes:
        x_min = int(cx - w / 2)
        y_min = int(cy - h / 2)
        boxes_nms.append([x_min, y_min, int(w), int(h)])

    indices = cv2.dnn.NMSBoxes(boxes_nms, confidences.tolist(), conf_thresh, iou_thresh)

    object_coords = {}
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            cls_id = class_ids[i]
            label = COCO_CLASSES.get(cls_id, None)
            if not label:
                continue
            
            cx, cy, w, h = boxes[i]
            cx_real = (cx / 480.0) * w_orig
            cy_real = (cy / 480.0) * h_orig
            w_px = (w / 480.0) * w_orig
            h_px = (h / 480.0) * h_orig
            
            if label not in object_coords:
                object_coords[label] = []
            object_coords[label].append({
                'center': (cx_real, cy_real),
                'width_px': w_px,
                'height_px': h_px
            })

    return object_coords

def process_desk_image(img, user_height, object_coords):
    h_img, w_img, _ = img.shape
    img_out = img.copy()

    FONT_SCALE = max(0.5, w_img / 1200.0)
    THICKNESS = max(1, int(FONT_SCALE * 1.5))

    x_origin, y_origin, w_origin_px = 0, 0, 0
    if 'laptop' in object_coords and len(object_coords['laptop']) > 0:
        x_origin, y_origin = object_coords['laptop'][0]['center']
        w_origin_px = object_coords['laptop'][0]['width_px']
    elif 'tvmonitor' in object_coords and len(object_coords['tvmonitor']) > 0:
        x_origin, y_origin = object_coords['tvmonitor'][0]['center']
        w_origin_px = object_coords['tvmonitor'][0]['width_px']
    else:
        x_origin, y_origin = w_img / 2, h_img * 0.4
        w_origin_px = w_img * 0.35

    if user_height <= 0:
        user_height = 170

    R_NORMAL_PX = w_origin_px * ((user_height * 0.24) / SCREEN_REAL_WIDTH_CM)
    R_MAX_PX = w_origin_px * ((user_height * 0.35) / SCREEN_REAL_WIDTH_CM)

    overlay = img_out.copy()
    cv2.circle(overlay, (int(x_origin), int(y_origin)), int(R_NORMAL_PX), (0, 255, 0), -1)
    cv2.circle(img_out, (int(x_origin), int(y_origin)), int(R_MAX_PX), (0, 0, 255), 2)
    cv2.addWeighted(overlay, 0.08, img_out, 0.92, 0, img_out)

    hazardous_keys = ['cup', 'bottle', 'wine glass']
    essential_keys = ['mouse', 'keyboard']
    document_keys = ['book']
    
    drawn_text_rects = []
    violations = []
    deductions = 0

    for label_name, object_list in object_coords.items():
        for obj in object_list:
            x_obj, y_obj = obj['center']
            w_obj = obj.get('width_px', 50)
            h_obj = obj.get('height_px', 50)

            if abs(x_obj - x_origin) < 20 and abs(y_obj - y_origin) < 20:
                continue

            distance_px = math.sqrt((x_obj - x_origin)**2 + (y_obj - y_origin)**2)
            if distance_px == 0:
                continue

            distance_cm = (distance_px / w_origin_px) * SCREEN_REAL_WIDTH_CM
            ux = (x_obj - x_origin) / distance_px
            uy = (y_obj - y_origin) / distance_px

            max_size = max(w_obj, h_obj)
            offset_dist = (max_size / 2) + 15

            text = ""
            color_text = (255, 255, 255)
            x_target, y_target = x_obj, y_obj

            if label_name in hazardous_keys and distance_px < R_MAX_PX:
                x_target = x_obj + (ux * offset_dist)
                y_target = y_obj + (uy * offset_dist)
                text = f"MOVE OUT ({distance_cm:.1f}cm)"
                color_text = (0, 0, 255)
                violations.append(f"**{label_name.title()}**: Placed too close to electronics ({distance_cm:.1f}cm)")
                deductions += 25

            elif label_name in essential_keys and distance_px > R_NORMAL_PX:
                x_target = x_obj - (ux * offset_dist)
                y_target = y_obj - (uy * offset_dist)
                text = f"PULL IN ({distance_cm:.1f}cm)"
                color_text = (0, 255, 255)
                violations.append(f"**{label_name.title()}**: Located outside natural reach zone ({distance_cm:.1f}cm)")
                deductions += 15

            elif label_name in document_keys:
                is_on_right_side = x_obj > x_origin
                if is_on_right_side and distance_px < R_MAX_PX:
                    x_target = x_obj + (ux * offset_dist)
                    y_target = y_obj + (uy * offset_dist)
                    text = "MOVE TO TOP-RIGHT (Mouse Blocked)"
                    color_text = (0, 165, 255)
                    violations.append("**Documents/Books**: Blocking mouse movement space on the right")
                    deductions += 15
                elif not is_on_right_side:
                    x_target, y_target = x_obj, y_obj
                    text = "USE DOCUMENT STAND (Save Neck)"
                    color_text = (255, 255, 0)
                    violations.append("**Documents/Books**: Placed flat on desk (may cause neck flexion)")
                    deductions += 10
                else:
                    x_target, y_target = x_obj, y_obj
                    text = "STACK & ALIGN (Keep Clean)"
                    color_text = (0, 255, 0)

            if text != "":
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
                offset_x = 10 if ux >= 0 else -text_width - 10
                offset_y = -10
                t_x = int(x_target) + int(offset_x)
                t_y = int(y_target) + int(offset_y)

                t_x = max(5, min(t_x, w_img - text_width - 5))
                t_y = max(text_height + 5, min(t_y, h_img - 5))

                r_x1, r_y1 = t_x - 4, t_y - text_height - 4
                r_x2, r_y2 = t_x + text_width + 4, t_y + baseline + 2

                is_overlapping = False
                for (b1, b2, b3, b4) in drawn_text_rects:
                    if not (r_x1 > b3 or r_x2 < b1 or r_y1 > b4 or r_y2 < b2):
                        is_overlapping = True
                        break

                if not is_overlapping:
                    drawn_text_rects.append((r_x1, r_y1, r_x2, r_y2))
                    cv2.rectangle(img_out, (r_x1, r_y1), (r_x2, r_y2), (0, 0, 0), cv2.FILLED)
                    cv2.putText(img_out, text, (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color_text, THICKNESS)

    score = max(0, 100 - deductions)
    if not violations:
        violations.append("Workspace setup is optimal with no ergonomic risks detected.")

    return img_out, score, violations

def process_single_image(img, user_height):
    object_coords = run_desk_onnx(img)
    annotated_img, score, violations = process_desk_image(img, user_height, object_coords)
    return annotated_img, violations, score

@app.get("/ping")
@app.get("/")
def read_root():
    return {"status": "online", "message": "ErgoAI Ultra-light ONNX Backend is awake & ready!"}

@app.post("/api/assess_desk")
async def assess_desk(
    file: UploadFile = File(...),
    user_height: float = Form(170.0),
    fatigue_level: int = Form(30)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return {"feedback": "Invalid image format.", "processed_image": "", "audio_base64": ""}

        h, w = img_bgr.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        annotated_bgr, violations, score = process_single_image(img_bgr, user_height)
        
        violations_text = "\n".join([f"* {v}" for v in violations]) if violations else "* Setup is optimal."
        spatial_section = f"### Spatial Alerts\n{violations_text}"

        final_report = f"### Ergonomic Score: **{score}/100**\n\n{spatial_section}"
        
        tts_raw_text = f"Ergonomic score is {score} out of 100. " + " ".join(violations)
        tts_text = clean_text_for_tts(tts_raw_text)
        audio_base64_str = get_voice_base64(tts_text, lang='en')

        return {
            "feedback": final_report,
            "processed_image": cv2_to_base64(annotated_bgr),
            "audio_base64": f"data:audio/mp3;base64,{audio_base64_str}" if audio_base64_str else ""
        }
    finally:
        gc.collect()

def run_pose_onnx(img):
    h_orig, w_orig = img.shape[:2]
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    session = get_pose_session()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})[0]

    out = outputs[0]
    if out.shape[0] < out.shape[1]: 
        out = out.T

    box_confs = out[:, 4]
    best_idx = np.argmax(box_confs)
    
    if box_confs[best_idx] < 0.05:
        return None, None

    best_pred = out[best_idx]
    kpts = best_pred[5:].reshape(17, 3)

    kpts[:, 0] = (kpts[:, 0] / 640.0) * w_orig
    kpts[:, 1] = (kpts[:, 1] / 640.0) * h_orig
    return kpts[:, :2], kpts[:, 2]

@app.post("/api/analyze_frame")
def analyze_frame(data: ImageData):
    try:
        try:
            encoded_data = data.image_base64.split(',')[1] if ',' in data.image_base64 else data.image_base64
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid Image Data")
        except Exception:
            return {"status": "Invalid Image", "is_bad_posture": False, "eye_dist": 0, "shoulder_y": 0, "is_success": False}

        kpts, confs = run_pose_onnx(img)
        
        status = "No person detected"
        current_eye_dist = 0.0
        current_shoulder_y = 0.0
        color = (0, 0, 255)
        is_success = False
        is_bad_posture = False

        if kpts is not None and len(kpts) >= 7:
            nose = kpts[0]
            left_eye, right_eye = kpts[1], kpts[2]
            left_shoulder, right_shoulder = kpts[5], kpts[6]
            CONF_THRESHOLD = 0.03

            if confs[1] > CONF_THRESHOLD and confs[2] > CONF_THRESHOLD:
                current_eye_dist = calculate_distance(left_eye, right_eye)
            elif confs[0] > CONF_THRESHOLD and confs[1] > CONF_THRESHOLD:
                current_eye_dist = calculate_distance(nose, left_eye) * 2
            elif confs[0] > CONF_THRESHOLD and confs[2] > CONF_THRESHOLD:
                current_eye_dist = calculate_distance(nose, right_eye) * 2

            if confs[5] > CONF_THRESHOLD and confs[6] > CONF_THRESHOLD:
                current_shoulder_y = float((left_shoulder[1] + right_shoulder[1]) / 2)
            elif confs[5] > CONF_THRESHOLD:
                current_shoulder_y = float(left_shoulder[1])
            elif confs[6] > CONF_THRESHOLD:
                current_shoulder_y = float(right_shoulder[1])
            elif confs[0] > CONF_THRESHOLD:
                current_shoulder_y = float(nose[1] + 100)

            if current_eye_dist > 0:
                shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) if (confs[5] > CONF_THRESHOLD and confs[6] > CONF_THRESHOLD) else 0

                if data.is_calibration:
                    if shoulder_tilt > 25:
                        status = "Error: SHOULDERS TILTED! Please sit straight."
                    else:
                        status = "Success: Standard posture saved!"
                        color = (0, 255, 0)
                        is_success = True
                else:
                    if data.baseline_eye_dist > 0 and current_eye_dist > (data.baseline_eye_dist * 1.25):
                        status = "Warning: Too close to screen!"
                        is_bad_posture = True
                    elif data.baseline_shoulder_y > 0 and current_shoulder_y > (data.baseline_shoulder_y + 20):
                        status = "Warning: Bad posture (Slouching)!"
                        is_bad_posture = True
                    elif shoulder_tilt > 20:
                        status = "Warning: Bad posture (Leaning)!"
                        is_bad_posture = True
                    else:
                        status = "Good posture"
                        color = (0, 255, 0)

            cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 4, (255, 0, 0), -1)
            cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 4, (255, 0, 0), -1)
            if confs[5] > CONF_THRESHOLD and confs[6] > CONF_THRESHOLD:
                cv2.line(img, (int(left_shoulder[0]), int(left_shoulder[1])), 
                         (int(right_shoulder[0]), int(right_shoulder[1])), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "image": f"data:image/jpeg;base64,{processed_base64}",
            "status": status,
            "is_bad_posture": is_bad_posture,
            "eye_dist": current_eye_dist,
            "shoulder_y": current_shoulder_y,
            "is_success": is_success
        }
    finally:
        gc.collect()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
