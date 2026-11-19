import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# =====================================================
# 1) ตั้งค่า
# =====================================================
MODEL_PATH = (r"D:/cow.v1i.yolov12/runs/segment/train/weights/best.pt")        # ไฟล์โมเดลที่เทรนไว้
CONF_THRES = 0.5
WIDTH, HEIGHT, FPS = 640, 480, 30

# สีกรอบข้อความของแต่ละสายพันธุ์ (BGR)
BREED_COLORS = {
    "brahman":   (0, 165, 255),   # ส้ม
    "charolais": (255, 200, 0),   # ฟ้า
}

# สัมประสิทธิ์สมการน้ำหนักของแต่ละสายพันธุ์
# *** แทนค่าด้วยตัวเลขที่ได้จากข้อมูลจริงของคุณ (fit จากน้ำหนักที่ชั่งจริง) ***
# weight = a + b*area + c*length + d*height
WEIGHT_COEFS = {
    "brahman":   dict(a=0.0, b=0.0, c=0.0, d=0.0),
    "charolais": dict(a=0.0, b=0.0, c=0.0, d=0.0),
}

# =====================================================
# 2) โหลดโมเดล
# =====================================================
model = YOLO(MODEL_PATH)
print("คลาสในโมเดล:", model.names)   # ตรวจสอบชื่อคลาสให้ตรงกับคีย์ด้านบน

# =====================================================
# 3) เปิดกล้อง LiDAR (Intel RealSense)
# =====================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intr.fx, intr.fy


# =====================================================
# 4) ฟังก์ชันช่วย
# =====================================================
def estimate_weight(breed_key, area_m2, length_m, height_m):
    """ประเมินน้ำหนัก (กก.) ตามสายพันธุ์"""
    c = WEIGHT_COEFS.get(breed_key)
    if c is None:
        return 0.0
    return c["a"] + c["b"] * area_m2 + c["c"] * length_m + c["d"] * height_m


def mask_to_features(mask, depth_m):
    """คำนวณพื้นที่/ความยาว/ความสูง (เมตร) และระยะจากกล้อง จาก mask + depth"""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    z = depth_m[ys, xs]
    valid = z > 0
    if valid.sum() < 100:
        return None
    xs, ys, z = xs[valid], ys[valid], z[valid]

    area_m2 = float(np.sum(z ** 2 / (fx * fy)))   # พื้นที่จริงของแต่ละพิกเซล ≈ z²/(fx*fy)
    dist_m = float(np.median(z))                  # ระยะห่างจากกล้อง
    length_m = (xs.max() - xs.min()) * dist_m / fx
    height_m = (ys.max() - ys.min()) * dist_m / fy
    return area_m2, length_m, height_m, dist_m


def draw_label(img, lines, x, y, color):
    """วาดข้อความหลายบรรทัดเหนือตัวโค (บรรทัดแรกอยู่บนสุด)"""
    line_h = 26
    top = max(y - line_h * len(lines), 5)
    for i, (text, scale) in enumerate(lines):
        pos = (x, top + (i + 1) * line_h - 6)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# =====================================================
# 5) ลูปหลัก
# =====================================================
try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

        results = model.predict(color, conf=CONF_THRES, verbose=False)[0]
        annotated = results.plot(labels=False, conf=False)   # วาดกรอบ/mask โดยไม่มีป้ายซ้ำซ้อน

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            for i, m in enumerate(masks):
                m = cv2.resize(m, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
                feats = mask_to_features(m, depth_m)
                if feats is None:
                    continue
                area, length, height, dist = feats

                # สายพันธุ์จากโมเดลที่เทรนไว้
                cls_id = int(results.boxes.cls[i])
                conf = float(results.boxes.conf[i])
                breed_key = model.names[cls_id].strip().lower()
                breed_name = breed_key.capitalize()

                weight = estimate_weight(breed_key, area, length, height)
                color_bgr = BREED_COLORS.get(breed_key, (0, 255, 0))

                x1, y1 = map(int, results.boxes.xyxy[i][:2].tolist())
                draw_label(
                    annotated,
                    [
                        (f"{breed_name} ({conf*100:.0f}%)", 0.7),
                        (f"Weight: {weight:.0f} kg", 0.8),
                        (f"Distance: {dist:.2f} m", 0.6),
                    ],
                    x1, y1, color_bgr,
                )

                print(f"{breed_name} | น้ำหนัก {weight:.1f} kg | ระยะ {dist:.2f} m")

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_m, alpha=60), cv2.COLORMAP_JET)
        cv2.imshow("Cattle weight estimation  (q = quit)",
                   np.hstack((annotated, depth_vis)))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()