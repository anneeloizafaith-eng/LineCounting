import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Line Counting", page_icon="👥")
st.title("Line Counting 👥")

model = YOLO(r'C:\Users\Angelica\OneDrive\Desktop\Line Counting\best.pt')

# Settings
cashier_count = st.number_input("How many cashiers are in the image?",
                                min_value=0, max_value=5, value=1)
cashier_side = st.radio("Where is the cashier?",
                        ["Left side", "Right side"])

# Both inputs showing at same time
st.subheader("📷 Take a Photo")
camera_file = st.camera_input("Use camera")

st.subheader("🖼️ Or Upload an Image")
upload_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Use whichever input is provided
img_file = camera_file if camera_file else upload_file

def process_image(img_file):
    img = np.array(Image.open(img_file))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img_bgr, conf=0.3)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    total_people = len(boxes)
    queue_count = total_people - cashier_count

    if cashier_side == "Left side":
        boxes_sorted = sorted(boxes, key=lambda x: x[0], reverse=False)
    else:
        boxes_sorted = sorted(boxes, key=lambda x: x[0], reverse=True)

    for i, box in enumerate(boxes_sorted):
        x1, y1, x2, y2 = map(int, box)
        if i < cashier_count:
            cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img_bgr, 'Cashier', (x1,y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        else:
            cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img_bgr, 'Queue', (x1,y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb)

    col1, col2 = st.columns(2)
    col1.metric("👥 Total People", total_people)
    col2.metric("🟢 People in Queue", max(0, queue_count))

if img_file:
    process_image(img_file)