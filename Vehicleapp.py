import cv2
import pytesseract
import re
import numpy as np
import sqlite3
import streamlit as st
import base64
import pywhatkit as kit
import datetime
import random
import pandas as pd  # For displaying database data in a table

# ✅ Streamlit Page Config
st.set_page_config(
    page_title="🚦 Automatic Number Plate Recognition (ANPR) System",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✅ Path to Tesseract OCR (Update if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\LENOVO\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# ✅ Load Haar Cascade for License Plate Detection
cascade_path = r'C:\Users\LENOVO\Documents\Naresh IT\PROJECT\NUMBER_PLATE_DETECTION\haarcascade_russian_plate_number (2).xml'
plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    st.error("Error: Haar cascade file not loaded correctly!")
    st.stop()

# ✅ Function to Set Background Image
def set_bg(image_file):
    with open(image_file, "rb") as image:
        base64_image = base64.b64encode(image.read()).decode()
    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# ✅ Apply Background
set_bg(r"C:\Users\LENOVO\Pictures\3254.jpg")

# ✅ Custom CSS for Animations and Styling
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        text-align: center;
        color: #2875A7;
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Streamlit UI
st.markdown('''
    <div style="border: 3px solid black; border-radius: 10px; padding: 20px; background: rgba(0, 0, 0, 0.7);">
        <h1 style="text-align:center; color:white;">
            🚦 Automatic Number Plate Recognition (ANPR) System
        </h1>
    </div>
''', unsafe_allow_html=True)


# ✅ Sidebar for Additional Options
with st.sidebar:
    st.markdown("### 🛠️ Settings")
    st.write("Configure your ANPR system here.")
    enable_whatsapp = st.checkbox("Enable WhatsApp Alerts", value=True)
    enable_database = st.checkbox("Enable Database Lookup", value=True)

    st.markdown("---")
    st.markdown("### 📊 Database Management")
    if st.button("View Database"):
        # Connect to the SQLite database
        conn = sqlite3.connect('vehicle_database.db')
        cursor = conn.cursor()

        # Fetch all data from the VehicleOwners table
        cursor.execute("SELECT * FROM VehicleOwners")
        rows = cursor.fetchall()

        # Close the connection
        conn.close()

        # Display the data in a table format
        if rows:
            st.markdown("### 📊 Vehicle Owners Database")
            st.write("Here is the data stored in the database:")
            
            # Convert the data to a pandas DataFrame for better visualization
            df = pd.DataFrame(rows, columns=["Vehicle Number", "Owner Name", "Registration Date", "Mobile Number"])
            st.dataframe(df)
        else:
            st.warning("⚠ No data found in the database!")

    st.markdown("---")
    st.markdown("### 📧 Contact Us")
    st.write("Report issues or suggest improvements:")
    contact_form = st.text_area("Your Message")
    if st.button("Submit"):
        st.success("Thank you for your feedback!")

# ✅ Function to Create Dummy Database
def create_dummy_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS VehicleOwners (
            VehicleNumber TEXT PRIMARY KEY,
            OwnerName TEXT,
            RegistrationDate TEXT,
            MobileNumber TEXT
        )
    ''')

    # Insert dummy data
    dummy_data = [
        ("KA01AB1234", "Alice Smith", "2023-01-15", "9876543210"),
        ("TN02CD5678", "Bob Johnson", "2022-08-20", "8765432109"),
        ("AP03EF9012", "Charlie Brown", "2024-03-10", "7654321098"),
        ("KL04GH3456", "Diana Miller", "2021-11-05", "6543210987"),
        ("MH05IJ7890", "Ethan White", "2023-05-25", "5432109876"),
        ("TS06KL1234", "Sophia Taylor", "2022-12-01", "4321098765"),
        ("GJ07MN5678", "David Garcia", "2024-02-18", "3210987654"),
        ("HR08OP9012", "Olivia Rodriguez", "2021-07-08", "2109876543"),
        ("SNK8338A", "James Wilson", "2023-09-30", "9556290318"),  
        ("UP10RS7890", "Emily Martinez", "2022-04-12", "9087654321"),
        ("MH12DE1433", "Liam Anderson", "2023-06-01", "8076543219"),
    ]

    cursor.executemany("INSERT OR REPLACE INTO VehicleOwners VALUES (?, ?, ?, ?)", dummy_data)
    conn.commit()
    conn.close()
    print("Dummy database created with table 'VehicleOwners'.")

# ✅ Create the database if it doesn't exist
database_file = "vehicle_database.db"
create_dummy_database(database_file)

# ✅ Upload Image File
st.markdown('''
    <h5 style="text-align:center; color:#FFFF00;">
        Upload Your Image (JPG, JPEG, PNG) Below to Start Detection! 📸
    </h5>
''', unsafe_allow_html=True)

uploaded_file = st.file_uploader("✨ Upload Your Image", type=["jpg", "jpeg", "png"])

# ✅ Function to Query the Database
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

# ✅ Function to Send WhatsApp Message
def send_whatsapp_message(owner_name, vehicle_number, mobile_number):
    locations = ["MG Road, Bangalore", "Anna Salai, Chennai", "Marine Drive, Mumbai", "Connaught Place, Delhi", "Banjara Hills, Hyderabad"]
    violation_location = random.choice(locations)

    now = datetime.datetime.now()
    date_time = now.strftime("%d-%m-%Y %I:%M %p")

    fine_amount = random.randint(500, 2000)
    due_date = (now + datetime.timedelta(days=15)).strftime("%d-%m-%Y")

    message = f"""
🚨 Traffic Violation Alert! 🚗💨

Dear Vehicle Owner,

Your vehicle **{vehicle_number}** was detected overspeeding at **{violation_location}** on **{date_time}** by the automated traffic monitoring system. Speeding is a serious offense and can endanger lives, including yours and others on the road.

🔴 **Challan Issued:** ₹{fine_amount}  
📍 **Violation Location:** {violation_location}  
📅 **Date & Time:** {date_time}  
⏳ **Due Date for Payment:** {due_date}  

💳 **How to Pay:**  
You can pay your fine online at https://echallan.parivahan.gov.in/ or visit your nearest traffic police station.  
Failure to pay by the due date may result in additional penalties, legal action, or restrictions on your vehicle registration and driving license.

🚦 **Drive Safe, Stay Safe!**  
Speed limits exist to protect everyone on the road. Avoid unnecessary fines and ensure a safe journey for yourself and others.

For any queries or to check pending challans, visit https://echallan.parivahan.gov.in/ or contact your nearest traffic police station.

- **Traffic Police 🚔**
    """

    hour = now.hour
    minute = now.minute + 2

    try:
        kit.sendwhatmsg(f"+91{mobile_number}", message, hour, minute)
        st.success(f"📩 WhatsApp alert sent to {owner_name} ({mobile_number})!")
    except Exception as e:
        st.error(f"❌ Error sending message: {e}")

# ✅ Process Uploaded Image
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting license plate..."):
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
        st.markdown(f'''
    <div style="border: 3px solid #28A745; border-radius: 10px; padding: 10px; background-color: #f0f8ff;">
        <p style="color:#28A745; font-size:24px; font-weight:bold; text-align:center;">
            ✅ Detected License Plate: <span style="color:#FF5733;">{plate_text}</span>
        </p>
    </div>
''', unsafe_allow_html=True)

        if enable_database:
            vehicle_data = query_database(plate_text)

            if vehicle_data:
                owner_name, registration_date, mobile_number = vehicle_data[1], vehicle_data[2], vehicle_data[3]

                if st.button("Show Vehicle Details"):
                    st.markdown(f"""
                        <div style="border: 3px solid #001F3F; border-radius: 10px; padding: 15px; background-color: #008080;">
                            <p><b>👤 Owner Name:</b> {owner_name}</p>
                            <p><b>📅 Registration Date:</b> {registration_date}</p>
                            <p><b>📞 Contact:</b> {mobile_number}</p>
                        </div>
                    """, unsafe_allow_html=True)

                if enable_whatsapp and st.button("Send WhatsApp Alert"):
                    send_whatsapp_message(owner_name, plate_text, mobile_number)
            else:
                st.warning("⚠ No matching record found in the database.")
        else:
            st.warning("⚠ Database lookup is disabled in settings.")
    else:
        st.warning("❌ No text detected from the license plate!")

# ✅ Footer
st.markdown("---")
st.markdown("""
    <div style="text-align:center; color:#6C757D;">
        <p>© 2025 Smart Traffic AI | Built with ❤️ by Shuvendu</p>
        <p> 
            <a href="https://www.linkedin.com/in/shuvendubarik" target="_blank">LinkedIn</a> | 
            <a href="https://github.com/shuvendu-git" target="_blank">GitHub</a>
        </p>
    </div>
""", unsafe_allow_html=True)
