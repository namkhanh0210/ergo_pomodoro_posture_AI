import base64
import io
import math
import os
import re
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

try:
    from google import genai
except ImportError:
    genai = None

app = FastAPI(title="ErgoAI - YOLOv8 System (Desk & Pose)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

desk_model = YOLO('yolov8n.pt') 
pose_model = YOLO('yolov8n-pose.pt')

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
gemini_client = None
if GEMINI_API_KEY and genai:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[Warning] Failed to initialize Gemini Client: {e}")

AUDIO_CACHE = {}
MAX_CACHE_SIZE = 100
SCREEN_REAL_WIDTH_CM = 35.0

class ImageData(BaseModel):
    image_base64: str
    is_calibration: bool
    baseline_eye_dist: float = 0.0
    baseline_shoulder_y: float = 0.0

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def cv2_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def clean_text_for_tts(text):
    cleaned = re.sub(r'[\#\*\_\`\>\-\[\]\(\)]', ' ', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def get_voice_base64(text: str, lang='en') -> str:
    if not text: 
        return ""
    
    # Dọn dẹp cache nếu quá đầy để tránh tràn RAM
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
        except Exception as e:
            print(f"[TTS Error]: {e}")
            return ""
    return AUDIO_CACHE[cache_key]

def process_single_image(img, user_height):
    h_img, w_img, _ = img.shape
    img_out = img.copy()

    results = desk_model(img_out, conf=0.20, imgsz=640, verbose=False)
    object_coords = {}
    for r in results:
        for box in r.boxes:
            label = desk_model.names[int(box.cls[0])].lower()
            x_center, y_center, width, height = box.xywh[0].tolist()
            if label not in object_coords:
                object_coords[label] = []
            object_coords[label].append({'center': (x_center, y_center), 'width_px': width})

    x_origin, y_origin, w_origin_px = 0, 0, 0
    if 'laptop' in object_coords:
        x_origin, y_origin = object_coords['laptop'][0]['center']
        w_origin_px = object_coords['laptop'][0]['width_px']
    elif 'tvmonitor' in object_coords:
        x_origin, y_origin = object_coords['tvmonitor'][0]['center']
        w_origin_px = object_coords['tvmonitor'][0]['width_px']
    else:
        x_origin, y_origin = w_img / 2, h_img * 0.4
        w_origin_px = w_img * 0.35

    raw_r_normal = w_origin_px * ((user_height * 0.24) / SCREEN_REAL_WIDTH_CM)
    raw_r_max = w_origin_px * ((user_height * 0.35) / SCREEN_REAL_WIDTH_CM)
    max_limit = min(w_img, h_img) * 0.35
    R_NORMAL_PX = int(min(raw_r_normal, max_limit))
    R_MAX_PX = int(min(raw_r_max, max_limit * 1.3))

    overlay = img_out.copy()
    cv2.circle(overlay, (int(x_origin), int(y_origin)), R_NORMAL_PX, (0, 255, 0), -1)
    cv2.circle(img_out, (int(x_origin), int(y_origin)), R_MAX_PX, (0, 0, 255), 3)
    cv2.addWeighted(overlay, 0.20, img_out, 0.80, 0, img_out)

    violations = []
    deductions = 0
    for label_name, object_list in object_coords.items():
        for obj in object_list:
            x_obj, y_obj = obj['center']
            if abs(x_obj - x_origin) < 20 and abs(y_obj - y_origin) < 20: 
                continue
            
            distance_px = math.sqrt((x_obj - x_origin)**2 + (y_obj - y_origin)**2)
            if distance_px == 0: 
                continue
            distance_cm = (distance_px / w_origin_px) * SCREEN_REAL_WIDTH_CM

            if label_name in ['cup', 'bottle'] and distance_px < R_MAX_PX:
                violations.append(f"**{label_name.title()}**: In danger zone ({distance_cm:.1f}cm)")
                deductions += 15
            elif label_name in ['mouse', 'keyboard'] and distance_px > R_NORMAL_PX:
                violations.append(f"**{label_name.title()}**: Too far ({distance_cm:.1f}cm)")
                deductions += 10

    score = max(0, 100 - deductions)
    return img_out, list(dict.fromkeys(violations)), score

@app.get("/")
def read_root():
    return {"status": "online", "message": "ErgoAI Production Backend is running!"}

@app.post("/api/assess_desk")
def assess_desk(
    file: UploadFile = File(...),
    user_height: float = Form(170.0),
    fatigue_level: int = Form(30),
    custom_notes: str = Form("")
):
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    annotated_bgr, violations, score = process_single_image(img_bgr, user_height)
    
    violations_text = "\n".join([f"* {v}" for v in violations]) if violations else "* Setup is optimal."
    spatial_section = f"### Spatial Alerts\n{violations_text}\n"

    gemini_insights = ""
    if gemini_client:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            raw_img = Image.fromarray(img_rgb)
            raw_img.thumbnail((1024, 1024))
            
            prompt = f"User Height: {user_height}cm. Score: {score}. Issues: {violations_text}. Be concise in English."
            response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, raw_img])
            gemini_insights = response.text if response.text else ""
        except Exception as err:
            gemini_insights = f"\n> AI Insight Error: `{str(err)}`"

    final_report = f"### Ergonomic Score: **{score}/100**\n\n{spatial_section}\n---\n{gemini_insights}"
    tts_text = clean_text_for_tts(gemini_insights if gemini_insights else "Analysis complete.")
    audio_base64_str = get_voice_base64(tts_text, lang='en')

    return {
        "feedback": final_report,
        "processed_image": cv2_to_base64(annotated_bgr),
        "audio_base64": f"data:audio/mp3;base64,{audio_base64_str}" if audio_base64_str else ""
    }
 
def analyze_frame(data: ImageData):
    try:
        encoded_data = data.image_base64.split(',')[1] if ',' in data.image_base64 else data.image_base64
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid Image Data")
    except Exception:
        return {"status": "Invalid Image", "is_bad_posture": False, "eye_dist": 0, "shoulder_y": 0, "is_success": False}

    results = pose_model(img, imgsz=320, verbose=False)
    
    status = "No person detected"
    current_eye_dist = 0.0
    current_shoulder_y = 0.0
    color = (0, 0, 255)
    is_success = False
    is_bad_posture = False

    if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        confs = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(len(kpts))
        
        CONF_THRESHOLD = 0.5

        if len(kpts) >= 7:
            left_eye, right_eye = kpts[1], kpts[2]
            left_shoulder, right_shoulder = kpts[5], kpts[6]

            if (confs[1] > CONF_THRESHOLD and confs[2] > CONF_THRESHOLD and
                confs[5] > CONF_THRESHOLD and confs[6] > CONF_THRESHOLD and
                np.any(left_eye) and np.any(right_eye) and 
                np.any(left_shoulder) and np.any(right_shoulder)):

                current_eye_dist = calculate_distance(left_eye, right_eye)
                current_shoulder_y = float((left_shoulder[1] + right_shoulder[1]) / 2)

                shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
                eye_tilt = abs(left_eye[1] - right_eye[1])

                if data.is_calibration:
                    if current_eye_dist > 60:
                        status = "Error: Too CLOSE! Move back 50-70cm and recalibrate."
                        color = (0, 0, 255)
                    elif current_eye_dist < 30:
                        status = "Error: Too FAR! Move closer 50-70cm and recalibrate."
                        color = (0, 0, 255)
                    elif shoulder_tilt > 12:
                        status = "Error: SHOULDERS TILTED! Please sit straight."
                        color = (0, 0, 255)
                    else:
                        status = "Success: Standard posture saved!"
                        color = (0, 255, 0)
                        is_success = True
                else:
                    if data.baseline_eye_dist > 0 and current_eye_dist > (data.baseline_eye_dist * 1.25):
                        status = "Warning: Too close to screen!"
                        is_bad_posture = True
                        color = (0, 0, 255)
                    elif data.baseline_shoulder_y > 0 and current_shoulder_y > (data.baseline_shoulder_y + 15):
                        status = "Warning: Bad posture (Slouching)!"
                        is_bad_posture = True
                        color = (0, 0, 255)
                    elif shoulder_tilt > 18 or eye_tilt > 15:
                        status = "Warning: Bad posture (Leaning)!"
                        is_bad_posture = True
                        color = (0, 0, 255)
                    else:
                        status = "Good posture"
                        is_bad_posture = False
                        color = (0, 255, 0)

                cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 4, (255, 0, 0), -1)
                cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 4, (255, 0, 0), -1)
                cv2.line(img, (int(left_shoulder[0]), int(left_shoulder[1])), 
                              (int(right_shoulder[0]), int(right_shoulder[1])), (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    processed_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "image": f"data:image/jpeg;base64,{processed_base64}",
        "status": status,
        "is_bad_posture": is_bad_posture,
        "eye_dist": current_eye_dist,
        "shoulder_y": current_shoulder_y,
        "is_success": is_success
    }
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
if __name__ == "__main__":
  uvicorn.main("app:app", host="0.0.0.0", port=7860, reload=True)