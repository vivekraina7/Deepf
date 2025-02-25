import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pickle
from deepface import DeepFace
from retinaface import RetinaFace
import base64
from PIL import Image
import hashlib
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
import json

# Set page config
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📸",
    layout="wide"
)


# Apply custom CSS
def local_css():
    st.markdown("""
    <style>
        /* Main color scheme */
        :root {
            --primary-color: #4CAF50;
            --secondary-color: #2196F3;
            --background-color: ;
            --text-color: ;
            --card-color: #ffffff;
        }
        
        /* Page background */
        .st-emotion-cache-a1j6ntc, .main, .stApp {
            background-color: var(--background-color);
        }
        
        /* Sidebar styling */
        .st-emotion-cache-6qob1r {
            background-color: #212121;
            color: white;
        }
        
        /* Cards */
        .css-card {
            border-radius: 10px;
            padding: 20px;
            background-color: var(--card-color);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 5px;
            border: none;
            padding: 8px 16px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #3d8b40;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Secondary buttons */
        .secondary-btn {
            background-color: var(--secondary-color) !important;
        }
        
        .secondary-btn:hover {
            background-color: #1976D2 !important;
        }
        
        /* Danger buttons */
        .danger-btn {
            background-color: #F44336 !important;
        }
        
        .danger-btn:hover {
            background-color: #D32F2F !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: var(--text-color);
            font-family: 'Segoe UI', sans-serif;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
            padding: 10px;
            border-bottom: 2px solid var(--primary-color);
        }
        
        h2 {
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        h3 {
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        
        /* Metrics */
        .metric-card {
            background-color: var(--card-color);
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--primary-color);
        }
        
        .metric-label {
            font-size: 1rem;
            color: #757575;
            margin-top: 5px;
        }
        
        /* Dataframes */
        .dataframe {
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        
        /* Login form styling */
        .login-form {
            max-width: 500px;
            margin: 0 auto;
            padding: 30px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        /* Attendance status indicators */
        .status-present {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .status-absent {
            color: #F44336;
            font-weight: bold;
        }
        
        /* Custom logo/header */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .app-logo {
            width: 60px;
            margin-right: 15px;
        }
        
        .app-header {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            margin: 0;
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: var(--primary-color);
        }
    </style>
    """, unsafe_allow_html=True)

# Apply CSS
local_css()

# Function to create necessary directories
def create_directories():
    dirs = ["database", "logs", "attendance", "users"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

# Initialize directories
create_directories()

# Constants
USER_DB_PATH = "users/users.json"
RESET_TOKENS_PATH = "users/reset_tokens.json"

# Initialize user database if it doesn't exist
if not os.path.exists(USER_DB_PATH):
    with open(USER_DB_PATH, 'w') as f:
        json.dump({}, f)

# Initialize reset tokens database if it doesn't exist
if not os.path.exists(RESET_TOKENS_PATH):
    with open(RESET_TOKENS_PATH, 'w') as f:
        json.dump({}, f)

# User Authentication Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    try:
        with open(USER_DB_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_users(users):
    with open(USER_DB_PATH, 'w') as f:
        json.dump(users, f)

def validate_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

def validate_password(password):
    # At least 8 characters, 1 uppercase, 1 lowercase, 1 digit
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True
    
def get_student_email(student_name):
    """Get email address for a student based on their name"""
    users = load_users()
    for email, user_data in users.items():
        # If the user is a student and their name matches
        if user_data.get('role') == 'Student' and user_data.get('full_name', '').lower() == student_name.lower():
            return email
    return None

def create_reset_token(email):
    token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    tokens = {}
    
    try:
        with open(RESET_TOKENS_PATH, 'r') as f:
            tokens = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        tokens = {}
    
    # Set token with expiration time (1 hour from now)
    tokens[token] = {
        "email": email,
        "expiry": (datetime.now().timestamp() + 3600)  # 1 hour expiry
    }
    
    with open(RESET_TOKENS_PATH, 'w') as f:
        json.dump(tokens, f)
    
    return token

def verify_reset_token(token):
    try:
        with open(RESET_TOKENS_PATH, 'r') as f:
            tokens = json.load(f)
            
        if token in tokens:
            token_data = tokens[token]
            current_time = datetime.now().timestamp()
            
            if current_time < token_data["expiry"]:
                return token_data["email"]
            else:
                # Token expired, remove it
                del tokens[token]
                with open(RESET_TOKENS_PATH, 'w') as f:
                    json.dump(tokens, f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    return None

def remove_reset_token(token):
    try:
        with open(RESET_TOKENS_PATH, 'r') as f:
            tokens = json.load(f)
        
        if token in tokens:
            del tokens[token]
            
            with open(RESET_TOKENS_PATH, 'w') as f:
                json.dump(tokens, f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

def send_password_reset_email(email, token):
    try:
        # This is a placeholder. In a production app, you would use actual SMTP credentials
        # For now, we'll just show the reset link in the UI
        reset_link = f"{token}"
        st.info(f"Password reset link (for demonstration only): {reset_link}")
        
        # In a real application, you would use the code below

        sender_email = "vivekraina33.vr@gmail.com"
        sender_password = "qalrcbjwgjnsbkcv"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = "Password Reset"
        
        body = f"Click the following link to reset your password: {reset_link}"
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

def send_absence_notification(student_email, student_name, subject, date):
    try:
        # This is a placeholder. In a production app, you would use actual SMTP credentials
        # For now, we'll just show a notification in the UI
        st.info(f"Absence notification would be sent to {student_email} (for demonstration only)")
        
        # In a real application, you would use the code below
        sender_email = "vivekraina33.vr@gmail.com"
        sender_password = "qalrcbjwgjnsbkcv"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = student_email
        msg['Subject'] = f"Absence Notification - {subject}"
        
        body = f'''Dear {student_name},
        
        This is to inform you that you were marked absent for {subject} on {date}.
        
        Please contact your instructor if you believe this is an error.
        
        Regards,
        Attendance System
        '''
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending absence notification: {e}")
        return False

# Attendance and Face Recognition Functions
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

def detect_faces(img):
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Detect faces
    try:
        faces = RetinaFace.detect_faces(img)
        face_list = []
        
        if isinstance(faces, dict):
            for key in faces:
                identity = faces[key]
                facial_area = identity["facial_area"]
                x1, y1, x2, y2 = facial_area
                face_img = img[y1:y2, x1:x2]
                face_list.append((face_img, facial_area))
        
        return face_list
    except Exception as e:
        st.error(f"Error in face detection: {e}")
        return []

def extract_embeddings(face_img):
    try:
        embedding = DeepFace.represent(face_img, model_name="Facenet512", enforce_detection=False)[0]
        return embedding
    except Exception as e:
        st.error(f"Error extracting embeddings: {e}")
        return None

def recognize_face(embedding, database, threshold=0.25):
    if not database:
        return "Unknown", 1.0
    
    min_distance = float('inf')
    identity = "Unknown"
    
    for name, stored_embeddings in database.items():
        for stored_emb in stored_embeddings:
            # Calculate cosine distance manually
            vector_a = np.array(embedding['embedding'])
            vector_b = np.array(stored_emb['embedding'])
            
            # Normalize vectors
            vector_a = vector_a / np.linalg.norm(vector_a)
            vector_b = vector_b / np.linalg.norm(vector_b)
            
            # Calculate cosine similarity and convert to distance
            similarity = np.dot(vector_a, vector_b)
            distance = 1 - similarity
            
            if distance < min_distance:
                min_distance = distance
                identity = name
    
    return (identity, min_distance) if min_distance < threshold else ("Unknown", min_distance)

def mark_attendance(name, subject):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Create subject directory if it doesn't exist
    subject_dir = f"attendance/{subject}"
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    
    attendance_file = f"{subject_dir}/{date}.csv"
    
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write("ID,Name,Time\n")
    
    df = pd.read_csv(attendance_file)
    
    # Check if the student is already marked for today
    if name not in df['Name'].values:
        id_num = len(df) + 1
        new_row = pd.DataFrame({'ID': [id_num], 'Name': [name], 'Time': [time_str]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        return True, "newly_marked"
    return False, "already_marked"

def draw_bounding_box(img, facial_area, name, distance, status="recognized"):
    x1, y1, x2, y2 = facial_area
    
    # Set color based on status
    if name == "Unknown":
        color = (0, 0, 255)  # Red for unknown
    elif status == "already_marked":
        color = (255, 165, 0)  # Orange for already marked students
    else:
        color = (0, 255, 0)  # Green for newly recognized students
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Display name and confidence
    confidence = (1 - distance) * 100 if distance <= 1 else 0
    label = f"{name} ({confidence:.1f}%)"
    if status == "already_marked":
        label += " (Already Marked)"
    
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return img

def get_available_subjects():
    attendance_dir = "attendance"
    if os.path.exists(attendance_dir):
        subjects = [d for d in os.listdir(attendance_dir) if os.path.isdir(os.path.join(attendance_dir, d))]
        return subjects
    return []

# Main App Interface
def main():
    # Session state initialization
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    if 'user_role' not in st.session_state:
        st.session_state.user_role = ""
    if 'show_login' not in st.session_state:
        st.session_state.show_login = True
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    if 'show_forgot_password' not in st.session_state:
        st.session_state.show_forgot_password = False
    if 'show_reset_password' not in st.session_state:
        st.session_state.show_reset_password = False
    if 'reset_token' not in st.session_state:
        st.session_state.reset_token = ""
        
    # Sidebar navigation for logged-in users
    if st.session_state.logged_in:
        with st.sidebar:
            st.title(f"Welcome, {st.session_state.user_email}")
            st.write(f"Role: {st.session_state.user_role}")
            
            # Display navigation based on user role
            if st.session_state.user_role == "Admin":
                page = st.radio("Go to", ["Register Student", "Take Attendance", "View Attendance Records"])
            else:
                page = "View Attendance Records"  # Students can only view attendance
                
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user_email = ""
                st.session_state.user_role = ""
                st.session_state.show_login = True
                st.experimental_rerun()
        
        # Page rendering based on selection
        if page == "Register Student" and st.session_state.user_role == "Admin":
            render_register_student_page()
        elif page == "Take Attendance" and st.session_state.user_role == "Admin":
            render_take_attendance_page()
        elif page == "View Attendance Records":
            render_view_attendance_page()
    else:
        # Authentication screens
        if st.session_state.show_login:
            render_login_page()
        elif st.session_state.show_signup:
            render_signup_page()
        elif st.session_state.show_forgot_password:
            render_forgot_password_page()
        elif st.session_state.show_reset_password:
            render_reset_password_page()

def render_login_page():
    st.title("Login")
    
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Login"):
            if not email or not password:
                st.error("Please enter your email and password.")
                return
            
            users = load_users()
            
            if email in users and users[email]['password'] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.user_role = users[email]['role']
                st.session_state.show_login = False
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid email or password. Please try again.")
    
    with col2:
        if st.button("Sign Up"):
            st.session_state.show_login = False
            st.session_state.show_signup = True
            st.experimental_rerun()
    
    with col3:
        if st.button("Forgot Password"):
            st.session_state.show_login = False
            st.session_state.show_forgot_password = True
            st.experimental_rerun()

def render_signup_page():
    st.title("Sign Up")
    
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
    role = st.selectbox("Role", ["Student", "Admin"])
    
    # Additional fields based on role
    if role == "Student":
        full_name = st.text_input("Full Name (must match attendance name)", key="full_name")
        student_id = st.text_input("Student ID", key="student_id")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create Account"):
            # Input validation
            if not email or not password or not confirm_password:
                st.error("Please fill out all fields.")
                return
            
            if role == "Student" and (not full_name or not student_id):
                st.error("Please fill out your full name and student ID.")
                return
            
            if not validate_email(email):
                st.error("Please enter a valid email address.")
                return
            
            if not validate_password(password):
                st.error("Password must be at least 8 characters with at least one uppercase letter, one lowercase letter, and one digit.")
                return
            
            if password != confirm_password:
                st.error("Passwords do not match.")
                return
            
            users = load_users()
            
            if email in users:
                st.error("An account with this email already exists.")
                return
            
            # Create new user
            user_data = {
                'password': hash_password(password),
                'role': role,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add student-specific information if applicable
            if role == "Student":
                user_data['full_name'] = full_name
                user_data['student_id'] = student_id
            
            users[email] = user_data
            save_users(users)
            
            st.success("Account created successfully! You can now login.")
            st.session_state.show_signup = False
            st.session_state.show_login = True
            st.experimental_rerun()
    
    with col2:
        if st.button("Back to Login"):
            st.session_state.show_signup = False
            st.session_state.show_login = True
            st.experimental_rerun()

def render_forgot_password_page():
    st.title("Forgot Password")
    
    email = st.text_input("Enter your email address", key="forgot_password_email")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Send Reset Link"):
            if not email:
                st.error("Please enter your email address.")
                return
            
            users = load_users()
            
            if email in users:
                # Create and send reset token
                token = create_reset_token(email)
                if send_password_reset_email(email, token):
                    st.success("Password reset link has been sent. Please check your email.")
                    # For demonstration purposes, we'll directly transition to the reset page
                    st.session_state.reset_token = token
                    st.session_state.show_forgot_password = False
                    st.session_state.show_reset_password = True
                    st.experimental_rerun()
            else:
                # Don't reveal if email exists or not for security reasons
                st.success("If your email is registered, you will receive a password reset link.")
    
    with col2:
        if st.button("Back to Login", key="back_to_login_from_forgot"):
            st.session_state.show_forgot_password = False
            st.session_state.show_login = True
            st.experimental_rerun()

def render_reset_password_page():
    st.title("Reset Password")
    
    token = st.session_state.reset_token
    email = verify_reset_token(token)
    
    if not email:
        st.error("Invalid or expired reset link. Please request a new one.")
        if st.button("Back to Forgot Password"):
            st.session_state.show_reset_password = False
            st.session_state.show_forgot_password = True
            st.experimental_rerun()
        return
    
    st.write(f"Reset password for: {email}")
    
    new_password = st.text_input("New Password", type="password", key="reset_new_password")
    confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
    
    if st.button("Reset Password"):
        if not new_password or not confirm_password:
            st.error("Please fill out all fields.")
            return
        
        if not validate_password(new_password):
            st.error("Password must be at least 8 characters with at least one uppercase letter, one lowercase letter, and one digit.")
            return
        
        if new_password != confirm_password:
            st.error("Passwords do not match.")
            return
        
        users = load_users()
        
        if email in users:
            users[email]['password'] = hash_password(new_password)
            save_users(users)
            
            # Remove used token
            remove_reset_token(token)
            
            st.success("Password has been reset successfully. You can now login with your new password.")
            st.session_state.show_reset_password = False
            st.session_state.show_login = True
            st.experimental_rerun()
        else:
            st.error("User not found. Please try again.")

def render_register_student_page():
    st.title("Register New Student")
    
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.text_input("Student ID", "")
        student_name = st.text_input("Student Name", "")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
            
            if st.button("Register with this image"):
                with st.spinner("Processing..."):
                    face_list = detect_faces(img)
                    
                    if face_list:
                        face_img, _ = face_list[0]  # Take the first detected face
                        embedding = extract_embeddings(face_img)
                        
                        if embedding:
                            # Save student data
                            database_file = "database/embeddings.pkl"
                            embeddings_db = load_embeddings(database_file)
                            
                            if student_name in embeddings_db:
                                embeddings_db[student_name].append(embedding)
                            else:
                                embeddings_db[student_name] = [embedding]
                            
                            save_embeddings(embeddings_db, database_file)
                            
                            # Save registration log
                            log_df = pd.DataFrame({
                                "ID": [student_id],
                                "Name": [student_name],
                                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                                "Method": ["Image Upload"],
                                "Registered_By": [st.session_state.user_email]
                            })
                            
                            if os.path.exists("logs/registration_log.csv"):
                                log_df.to_csv("logs/registration_log.csv", mode='a', header=False, index=False)
                            else:
                                log_df.to_csv("logs/registration_log.csv", index=False)
                            
                            st.success(f"Successfully registered {student_name}!")
                        else:
                            st.error("Failed to extract face embedding. Please try another image.")
                    else:
                        st.error("No face detected in the image. Please try another image.")
    
    with col2:
        st.subheader("Registered Students")
        database_file = "database/embeddings.pkl"
        embeddings_db = load_embeddings(database_file)
        
        if embeddings_db:
            student_list = []
            for name, embeddings in embeddings_db.items():
                student_list.append({"Name": name, "Samples": len(embeddings)})
            
            st.dataframe(pd.DataFrame(student_list), use_container_width=True)
            
            if st.button("Delete All Registrations"):
                if os.path.exists(database_file):
                    os.remove(database_file)
                    st.success("All registrations have been deleted.")
                    st.experimental_rerun()
                    
            # Option to delete specific student
            if student_list:
                student_to_delete = st.selectbox("Select student to delete", [student["Name"] for student in student_list])
                if st.button("Delete Selected Student"):
                    if student_to_delete in embeddings_db:
                        del embeddings_db[student_to_delete]
                        save_embeddings(embeddings_db, database_file)
                        st.success(f"{student_to_delete} has been deleted.")
                        st.experimental_rerun()
        else:
            st.info("No students registered yet.")

def render_take_attendance_page():
    st.title("Take Attendance")
    
    # Subject selection/creation
    subject_options = get_available_subjects()
    subject_options.insert(0, "Create New Subject")
    
    subject_selection = st.selectbox("Select Subject", subject_options)
    
    if subject_selection == "Create New Subject":
        new_subject = st.text_input("Enter New Subject Name")
        if new_subject:
            subject = new_subject
        else:
            st.warning("Please enter a subject name or select an existing subject.")
            subject = None
    else:
        subject = subject_selection
    
    if subject:
        # Add attendance date option (default to today)
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_date = st.date_input("Attendance Date", value=datetime.now())
        formatted_date = attendance_date.strftime("%Y-%m-%d")
        
        # Option to send absence notifications
        send_absence_emails = st.checkbox("Send absence notifications to students", value=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Upload class image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Load the database
                database_file = "database/embeddings.pkl"
                embeddings_db = load_embeddings(database_file)
                
                if not embeddings_db:
                    st.error("No students registered in the database. Please register students first.")
                else:
                    # Read the image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, 1)
                    
                    # Display original image
                    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
                    
                    # Process button
                    if st.button("Process Attendance"):
                        with st.spinner("Processing attendance..."):
                            # Detect faces
                            face_list = detect_faces(img)
                            img_with_boxes = img.copy()
                            
                            newly_marked_students = []
                            already_marked_students = []
                            unknown_faces = 0
                            
                            if face_list:
                                for face_img, facial_area in face_list:
                                    # Extract embedding
                                    embedding = extract_embeddings(face_img)
                                    
                                    if embedding:
                                        # Recognize face
                                        name, distance = recognize_face(embedding, embeddings_db)
                                        
                                        # Mark attendance if recognized
                                        status = "unknown"
                                        if name != "Unknown":
                                            marked, mark_status = mark_attendance(name, subject)
                                            status = mark_status
                                            
                                            if mark_status == "newly_marked":
                                                newly_marked_students.append(name)
                                            else:
                                                already_marked_students.append(name)
                                        else:
                                            unknown_faces += 1
                                        
                                        # Draw bounding box
                                        img_with_boxes = draw_bounding_box(img_with_boxes, facial_area, name, distance, status)
                                
                                # Display processed image
                                st.image(img_with_boxes, channels="BGR", caption="Processed Image", use_column_width=True)
                                
                                # Display recognition results
                                if newly_marked_students or already_marked_students:
                                    if newly_marked_students:
                                        st.success(f"Newly marked attendance for: {', '.join(newly_marked_students)}")
                                    if already_marked_students:
                                        st.info(f"Already marked today: {', '.join(already_marked_students)}")
                                    if unknown_faces > 0:
                                        st.warning(f"{unknown_faces} unknown face(s) detected.")
                                    
                                    # Send absence notifications if requested
                                    if send_absence_emails:
                                        # Get all registered students
                                        all_students = list(embeddings_db.keys())
                                        present_students = newly_marked_students + already_marked_students
                                        absent_students = [s for s in all_students if s not in present_students]
                                        
                                        # Notify absent students
                                        notifications_sent = 0
                                        for absent_student in absent_students:
                                            student_email = get_student_email(absent_student)
                                            if student_email:
                                                send_absence_notification(student_email, absent_student, subject, formatted_date)
                                                notifications_sent += 1
                                        
                                        if notifications_sent > 0:
                                            st.success(f"Absence notifications sent to {notifications_sent} students")
                                else:
                                    if unknown_faces > 0:
                                        st.warning(f"No registered students recognized. {unknown_faces} unknown face(s) detected.")
                                    else:
                                        st.warning("No faces detected in the image.")
                            else:
                                st.error("No faces detected in the image.")
        
        with col2:
            st.subheader(f"Today's Attendance - {subject}")
            
            today = datetime.now().strftime("%Y-%m-%d")
            attendance_file = f"attendance/{subject}/{today}.csv"
            
            if os.path.exists(attendance_file):
                df = pd.read_csv(attendance_file)
                st.dataframe(df, use_container_width=True)
                
                # Option to manually add a student
                st.subheader("Manually Add Attendance")
                registered_students = list(load_embeddings("database/embeddings.pkl").keys())
                if registered_students:
                    manual_student = st.selectbox("Select Student", registered_students)
                    if st.button("Add to Attendance"):
                        marked, status = mark_attendance(manual_student, subject)
                        if status == "newly_marked":
                            st.success(f"Added {manual_student} to today's attendance.")
                            st.experimental_rerun()
                        else:
                            st.info(f"{manual_student} was already marked for today.")
                
                # Download attendance
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{subject}_{today}.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No attendance records for today yet.")
                
                # Option to manually add a student even when no records exist
                st.subheader("Manually Add Attendance")
                registered_students = list(load_embeddings("database/embeddings.pkl").keys())
                if registered_students:
                    manual_student = st.selectbox("Select Student", registered_students)
                    if st.button("Add to Attendance"):
                        marked, status = mark_attendance(manual_student, subject)
                        st.success(f"Added {manual_student} to today's attendance.")
                        st.experimental_rerun()

def render_view_attendance_page():
    st.title("Attendance Records")
    
    # View type selection
    view_type = st.radio("View by", ["Subject", "Student"])
    
    if view_type == "Subject":
        render_subject_view()
    else:
        render_student_view()

def render_subject_view():
    """Display attendance records organized by subject"""
    # Subject selection
    subject_options = get_available_subjects()
    
    if not subject_options:
        st.info("No attendance records found for any subject.")
        return
    
    selected_subject = st.selectbox("Select Subject", subject_options)
    
    # List available attendance files for the selected subject
    attendance_dir = f"attendance/{selected_subject}"
    if os.path.exists(attendance_dir):
        attendance_files = [f for f in os.listdir(attendance_dir) if f.endswith('.csv')]
        
        if attendance_files:
            selected_date = st.selectbox("Select Date", sorted(attendance_files, reverse=True))
            
            if selected_date:
                attendance_path = os.path.join(attendance_dir, selected_date)
                df = pd.read_csv(attendance_path)
                
                # Show statistics
                total_students = len(df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Attendance", total_students)
                with col2:
                    st.metric("Subject", selected_subject)
                with col3:
                    st.metric("Date", selected_date.replace(".csv", ""))
                
                st.subheader("Attendance List")
                st.dataframe(df, use_container_width=True)
                
                # Admin-only options
                if st.session_state.user_role == "Admin":
                    # Option to delete a record
                    if not df.empty:
                        student_to_delete = st.selectbox("Select student to remove from attendance", df['Name'].tolist())
                        if st.button("Remove Student from Attendance"):
                            df = df[df['Name'] != student_to_delete]
                            df.to_csv(attendance_path, index=False)
                            st.success(f"Removed {student_to_delete} from attendance records.")
                            st.experimental_rerun()
                        
                        # Option to send absence notifications
                        if st.button("Send Absence Notifications"):
                            # Get all registered students
                            embeddings_db = load_embeddings("database/embeddings.pkl")
                            all_students = list(embeddings_db.keys())
                            present_students = df['Name'].tolist()
                            absent_students = [s for s in all_students if s not in present_students]
                            
                            # Notify absent students
                            notifications_sent = 0
                            for absent_student in absent_students:
                                student_email = get_student_email(absent_student)
                                if student_email:
                                    send_absence_notification(student_email, absent_student, selected_subject, selected_date.replace(".csv", ""))
                                    notifications_sent += 1
                            
                            if notifications_sent > 0:
                                st.success(f"Absence notifications sent to {notifications_sent} students")
                            else:
                                st.info("No absence notifications were sent")
                
                # Download attendance
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{selected_subject}_{selected_date}">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Visualization option
                st.subheader("Time Distribution")
                
                if 'Time' in df.columns and not df.empty:
                    try:
                        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
                        hour_counts = df['Hour'].value_counts().sort_index()
                        st.bar_chart(hour_counts)
                    except Exception as e:
                        st.error(f"Error creating visualization: {e}")
        else:
            st.info(f"No attendance records found for {selected_subject}.")
    else:
        st.info(f"No attendance records found for {selected_subject}.")

def render_student_view():
    """Display attendance records organized by student"""
    # Get all students with attendance records
    database_file = "database/embeddings.pkl"
    embeddings_db = load_embeddings(database_file)
    
    if not embeddings_db:
        st.info("No students registered in the database.")
        return
    
    # Get all registered students
    registered_students = list(embeddings_db.keys())
    
    # Select a student
    selected_student = st.selectbox("Select Student", registered_students)
    
    if selected_student:
        # Find all attendance records for this student across all subjects
        subject_options = get_available_subjects()
        
        if not subject_options:
            st.info("No attendance records found.")
            return
        
        # Create a list to store all attendance records
        all_attendance = []
        
        # Search through all subjects
        for subject in subject_options:
            attendance_dir = f"attendance/{subject}"
            if os.path.exists(attendance_dir):
                attendance_files = [f for f in os.listdir(attendance_dir) if f.endswith('.csv')]
                
                for date_file in attendance_files:
                    attendance_path = os.path.join(attendance_dir, date_file)
                    df = pd.read_csv(attendance_path)
                    
                    # Check if the student is in this attendance record
                    if selected_student in df['Name'].values:
                        # Get the student's record
                        student_record = df[df['Name'] == selected_student].iloc[0]
                        
                        # Add to the list
                        all_attendance.append({
                            'Subject': subject,
                            'Date': date_file.replace(".csv", ""),
                            'Time': student_record['Time']
                        })
        
        if all_attendance:
            # Convert to DataFrame
            attendance_df = pd.DataFrame(all_attendance)
            
            # Sort by date (most recent first)
            attendance_df = attendance_df.sort_values(by='Date', ascending=False)
            
            # Display attendance statistics
            st.subheader(f"Attendance Summary for {selected_student}")
            
            total_classes = len(attendance_df)
            unique_subjects = attendance_df['Subject'].nunique()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Classes Attended", total_classes)
            with col2:
                st.metric("Number of Subjects", unique_subjects)
            
            # Display detailed attendance
            st.subheader("Attendance Details")
            st.dataframe(attendance_df, use_container_width=True)
            
# Call the main function
if __name__ == "__main__":
    main()
