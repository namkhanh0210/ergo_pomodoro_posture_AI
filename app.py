import math
import cv2
import numpy as np

SCREEN_REAL_WIDTH_CM = 35.0

def process_desk_image(img, user_height, object_coords):
    """
    Evaluates workspace ergonomics:
    - Draws transparent Reach Zone circles
    - Renders warning labels without arrows
    - Calculates Ergonomic Score and generates English violation details
    """
    h_img, w_img, _ = img.shape
    img_out = img.copy()

    FONT_SCALE = max(0.5, w_img / 1200.0)
    THICKNESS = max(1, int(FONT_SCALE * 1.5))

    # 1. Determine Anchor Point (Laptop or Monitor)
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

    # 2. Calculate Reach Zones
    if user_height <= 0:
        user_height = 170

    R_NORMAL_PX = w_origin_px * ((user_height * 0.24) / SCREEN_REAL_WIDTH_CM)
    R_MAX_PX = w_origin_px * ((user_height * 0.35) / SCREEN_REAL_WIDTH_CM)

    # Draw Reach Zone Overlay
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

    # 3. Evaluate Objects and Distance Metrics
    for label_name, object_list in object_coords.items():
        for obj in object_list:
            x_obj, y_obj = obj['center']
            w_obj, h_obj = obj['width_px'], obj['height_px']

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

            # A. Liquids / Hazard Items in Danger Zone
            if label_name in hazardous_keys and distance_px < R_MAX_PX:
                x_target = x_obj + (ux * offset_dist)
                y_target = y_obj + (uy * offset_dist)
                text = f"MOVE OUT ({distance_cm:.1f}cm)"
                color_text = (0, 0, 255)
                violations.append(f"**{label_name.title()}**: Placed too close to electronics ({distance_cm:.1f}cm)")
                deductions += 25

            # B. Essential Items Outside Reach Zone
            elif label_name in essential_keys and distance_px > R_NORMAL_PX:
                x_target = x_obj - (ux * offset_dist)
                y_target = y_obj - (uy * offset_dist)
                text = f"PULL IN ({distance_cm:.1f}cm)"
                color_text = (0, 255, 255)
                violations.append(f"**{label_name.title()}**: Located outside natural reach zone ({distance_cm:.1f}cm)")
                deductions += 15

            # C. Books / Documents Clutter
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

            # 4. Render Label Overlays (No Arrows)
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

    # 5. Final Score Calculation
    score = max(0, 100 - deductions)
    if not violations:
        violations.append("Workspace setup is optimal with no ergonomic risks detected.")

    return img_out, score, violations
