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

# Set page config
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📸",
    layout="wide"
)

# Function to create necessary directories
def create_directories():
    dirs = ["database", "logs", "attendance"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

# Initialize directories
create_directories()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Register Student", "Take Attendance", "View Attendance Records"])

# Function to save embeddings
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Function to load embeddings
def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

# Function to detect faces using RetinaFace
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

# Function to extract face embeddings
def extract_embeddings(face_img):
    try:
        embedding = DeepFace.represent(face_img, model_name="Facenet512", enforce_detection=False)[0]
        return embedding
    except Exception as e:
        st.error(f"Error extracting embeddings: {e}")
        return None

# Function to recognize face
def recognize_face(embedding, database, threshold=0.4):
    if not database:
        return "Unknown", 1.0
    
    min_distance = float('inf')
    identity = "Unknown"
    
    for name, stored_embeddings in database.items():
        for stored_emb in stored_embeddings:
            # Calculate cosine distance manually instead of using DeepFace.verify
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

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    attendance_file = f"attendance/{date}.csv"
    
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

# Function to draw bounding boxes
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

# Register Student Page
if page == "Register Student":
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
                                "Method": ["Image Upload"]
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

# Take Attendance Page
elif page == "Take Attendance":
    st.title("Take Attendance")
    
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
                                        marked, mark_status = mark_attendance(name)
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
                            else:
                                if unknown_faces > 0:
                                    st.warning(f"No registered students recognized. {unknown_faces} unknown face(s) detected.")
                                else:
                                    st.warning("No faces detected in the image.")
                        else:
                            st.error("No faces detected in the image.")
    
    with col2:
        st.subheader("Today's Attendance")
        
        today = datetime.now().strftime("%Y-%m-%d")
        attendance_file = f"attendance/{today}.csv"
        
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            st.dataframe(df, use_container_width=True)
            
            # Option to manually add a student
            st.subheader("Manually Add Attendance")
            registered_students = list(load_embeddings("database/embeddings.pkl").keys())
            if registered_students:
                manual_student = st.selectbox("Select Student", registered_students)
                if st.button("Add to Attendance"):
                    marked, status = mark_attendance(manual_student)
                    if status == "newly_marked":
                        st.success(f"Added {manual_student} to today's attendance.")
                        st.experimental_rerun()
                    else:
                        st.info(f"{manual_student} was already marked for today.")
            
            # Download attendance
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="attendance_{today}.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No attendance records for today yet.")
            
            # Option to manually add a student even when no records exist
            st.subheader("Manually Add Attendance")
            registered_students = list(load_embeddings("database/embeddings.pkl").keys())
            if registered_students:
                manual_student = st.selectbox("Select Student", registered_students)
                if st.button("Add to Attendance"):
                    marked, status = mark_attendance(manual_student)
                    st.success(f"Added {manual_student} to today's attendance.")
                    st.experimental_rerun()

# View Attendance Records Page
elif page == "View Attendance Records":
    st.title("Attendance Records")
    
    # List available attendance files
    attendance_dir = "attendance"
    if os.path.exists(attendance_dir):
        attendance_files = [f for f in os.listdir(attendance_dir) if f.endswith('.csv')]
        
        if attendance_files:
            selected_date = st.selectbox("Select Date", sorted(attendance_files, reverse=True))
            
            if selected_date:
                attendance_path = os.path.join(attendance_dir, selected_date)
                df = pd.read_csv(attendance_path)
                
                # Show statistics
                total_students = len(df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Attendance", total_students)
                
                st.subheader("Attendance List")
                st.dataframe(df, use_container_width=True)
                
                # Option to delete a record
                if not df.empty:
                    student_to_delete = st.selectbox("Select student to remove from attendance", df['Name'].tolist())
                    if st.button("Remove Student from Attendance"):
                        df = df[df['Name'] != student_to_delete]
                        df.to_csv(attendance_path, index=False)
                        st.success(f"Removed {student_to_delete} from attendance records.")
                        st.experimental_rerun()
                
                # Download attendance
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{selected_date}">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Visualization option
                st.subheader("Time Distribution")
                
                if 'Time' in df.columns and not df.empty:
                    # Convert time strings to datetime objects for visualization
                    try:
                        df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
                        hour_counts = df['Hour'].value_counts().sort_index()
                        st.bar_chart(hour_counts)
                    except Exception as e:
                        st.error(f"Error creating visualization: {e}")
        else:
            st.info("No attendance records found.")
    else:
        st.info("No attendance records found.")

# Add footer
st.markdown("---")
st.markdown("Face Recognition Attendance System | Built with Streamlit, DeepFace, and RetinaFace")