import cv2
import pytesseract
import re
import numpy as np
import sqlite3
import streamlit as st

# Path to Tesseract executable (update this path as per your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\LENOVO\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load the Haar cascade for Russian license plate detection
cascade_path = r'C:\Users\LENOVO\Documents\Naresh IT\PROJECT\NUMBER_PLATE_DETECTION\haarcascade_russian_plate_number (2).xml'
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade file is loaded correctly
if plate_cascade.empty():
    st.error("Error: Haar cascade file not loaded correctly!")
    st.stop()

# Function to query the database
def query_database(plate_number):
    try:
        conn = sqlite3.connect('vehicle_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM VehicleOwners WHERE VehicleNumber = ?", (plate_number,))
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="License Plate Detection", page_icon="🚗", layout="centered")
st.markdown(
    """
    <style>
    .title {
        font-size: 35px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
    }
    .subtitle {
        font-size: 20px;
        color: #34495E;
        text-align: center;
    }
    .footer {
        font-size: 15px;
        color: #AAB7B8;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">🚘 Automatic Number Plate Recognition (ANPR) System</div>', unsafe_allow_html=True)
st.markdown(""" 
<div class="subtitle" style="text-align: center; font-size: 20px; color: #4CAF50; font-weight: bold;">
    📸 Snap it, upload it, and let us handle the rest! Detect license plates instantly and get the details you need.
</div>
""", unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader("✨ Upload Your Image (JPG, JPEG, PNG) and Let’s Detect License Plates! 📸", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    plates = plate_cascade.detectMultiScale(thresholded, scaleFactor=1.1, minNeighbors=4)
    plate_text = ""

    for (x, y, w, h) in plates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        plate_region = image[y:y + h, x:x + w]
        raw_text = pytesseract.image_to_string(plate_region, config='--oem 1 --psm 7')
        plate_text = re.sub(r'[^A-Za-z0-9]', '', raw_text).strip()

        if plate_text:
            cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(image, caption="🔍 License Plate Detection with OCR", use_column_width=True)

    if plate_text:
        st.success(f"✅ Detected License Plate: **{plate_text}**")
        vehicle_data = query_database(plate_text)

        if vehicle_data:
            owner_name, registration_date, mobile_number = vehicle_data[1], vehicle_data[2], vehicle_data[3]

            # Show details button
            if st.button("Show Vehicle Details"):
                st.markdown("### Vehicle Details")
                st.info(f"""
                **🚘 Owner Name:** {owner_name}  
                **📅 Registration Date:** {registration_date}  
                **📞 Mobile Number:** {mobile_number}
                """)
        else:
            st.warning("⚠️ No matching record found in the database.")
    else:
        st.warning("❌ No text detected from the license plate!")

st.markdown('<div class="footer">© 2025 License Plate Detector | Built with ❤️ by Shuvendu</div>', unsafe_allow_html=True)