import json
import base64
import os, time
import logging
import threading
import flet as ft
import numpy as np
import cv2, cvzone
from pages.utils import FaceClassifier, FaceDetector, center_crop_frame, update_attendance, DataCipher


class SignInPage(ft.UserControl):
    def __init__(self, page, camera_manager):
        super().__init__()
        self.page = page
        self.camera_manager = camera_manager
        self.file_data_path = 'assets/application_data/application_storage/registered_data.json'
        self.camera = self.camera_manager.get_camera() # Get shared camera instance
        self.face_detector = FaceDetector()
        self.data_cipher = DataCipher()
        self.face_classifier = FaceClassifier()
        self.running = True
        self.img = ft.Image(
            border_radius=ft.border_radius.all(20),
            width=400,
            height=400
        )
        self.prog_bar = ft.ProgressBar(
            visible=False,
            width=400,
            color=ft.colors.BLUE_300,
            bgcolor='#d3d3d3'
        )
        self.progress_status = ft.Text(
            "",
            visible=False,
            size=16,
            color=ft.colors.GREEN_800,
            text_align=ft.TextAlign.CENTER,
            weight=ft.FontWeight.BOLD
        )

        self.signin_button =  ft.Row(
            controls=[
                ft.Container(
                    border_radius=5,
                    expand=True,
                    bgcolor='#3b82f6',
                    gradient=ft.LinearGradient(
                        colors=['#bbf7d0', '#86efac', '#3b82f6'],
                    ),
                    content=ft.Text('Take Attendance and Grant Access', text_align=ft.TextAlign.CENTER, size=25),
                    padding=ft.padding.only(left=170, right=170, top=10, bottom=10),
                    on_click=self.sign_in,
                )
            ],
            alignment='center',
            vertical_alignment='center'
        )

    # Configure the logger
    logging.basicConfig(
        level=logging.ERROR,  # Set the log level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
        filename="error_log.log",  # Save logs to a file (optional)
        filemode="a"  # Append to the log file
    )

    logger = logging.getLogger(__name__)

    def _set_processing_state(self, processing: bool, message: str = ""):
        """Helper to synchronize progress bar and status text visibility"""
        self.prog_bar.visible = processing
        self.progress_status.visible = processing
        self.progress_status.value = message
        self.update()

    def did_mount(self):
        self.update_frame()

    def will_unmount(self):
        self.running = False
        if self.camera is not None:
            self.camera_manager.release_camera() # Release camera when unmounting

    def update_frame(self):
        def update():
            failure_count = 0  # Track consecutive failures
            max_retries = 5    # Maximum retries before stopping the thread
            while self.running:
                ret, frame = self.camera.read()
                if ret:
                    # Reset failure count on success
                    failure_count = 0

                    # detect face in frame and draw bounding box
                    face_loc = self.face_detector.detect_face(frame)
                    if face_loc:
                        x, y, w, h = face_loc
                        cvzone.cornerRect(frame, (x, y, w, h), l=30, t=5, colorR=(0,255,0))

                    # Crop and encode frame
                    cropped_frame = center_crop_frame(frame)
                    _, im_arr = cv2.imencode('.png', cropped_frame)
                    im_b64 = base64.b64encode(im_arr)
                    self.img.src_base64 = im_b64.decode("utf-8")
                    self.update()
                else:
                    # Increment failure count and log
                    failure_count += 1
                    self.show_snackbar(f"Webcam read failed ({failure_count}/{max_retries}). Retrying...")

                    # Stop the thread if max retries are reached
                    if failure_count >= max_retries:
                        # Only navigate back if still on the SignIn page
                        if self.page.route == '/sign_in':  
                            self.show_snackbar("Maximum retries reached. Stopping webcam feed. Navigating back to landing page.")
                            self.running = False
                            self.page.go('/')
                        break

                    time.sleep(1)  # Delay for retries
        # Start the thread
        threading.Thread(target=update, daemon=True).start()

    def build(self):
        return ft.Column(
            [
                ft.Divider(height=10, color='transparent'),
                ft.Text('Welcome to the SignIn Page', size=24, weight=ft.FontWeight.BOLD, text_align='center'),
                ft.Divider(height=20, color='transparent'),
                self.img,
                self.progress_status,
                self.prog_bar,
                ft.Divider(height=20, color='transparent'),
                self.signin_button
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

    def show_snackbar(self, message):
        """Display a snackbar with a message using the new Flet method."""
        snackbar = ft.SnackBar(
            bgcolor=ft.colors.GREY_900,
            content=ft.Text(message, color=ft.colors.WHITE)
        )
        # Append the snackbar to the page's overlay and make it visible
        self.page.overlay.append(snackbar)
        snackbar.open = True
        self.page.update()  # Refresh the page to show the snackbar

    def sign_in(self, e=None):
        try:
            # Show processing state
            self._set_processing_state(True, "Processing face...")

            # Capture and validate frame
            ret, frame = self.camera.read()
            if not ret or frame is None:
                self._set_processing_state(False)
                self.show_snackbar('Camera error, failed to capture image. Please try again.')
                return

            # Detect and crop face
            face_location = self.face_detector.detect_face(frame)
            if not face_location:
                self._set_processing_state(False)
                self.show_snackbar("No face detected. Please position your face properly.")
                return

            x, y, width, height = face_location
            cropped_face = frame[y:y+height, x:x+width]
            if cropped_face.size == 0:
                self._set_processing_state(False)
                self.show_snackbar("Unable to crop the face. Please try again.")
                return

            # Check for registered users
            if not os.path.exists(self.file_data_path):
                self._set_processing_state(False)
                self.show_snackbar("No registered users found. Please sign up first.")
                return

            try:
                with open(self.file_data_path, 'r') as f:
                    registered_users = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Error reading user data: {str(e)}")
                self._set_processing_state(False)
                self.show_snackbar("Error accessing user database. Please try again.")
                return

            # Recognize face
            recognized_email, confidence = self.face_classifier.recognize_face(cropped_face)
            print(f"Recognized Email: {recognized_email}")
            print(f"Confidence: {confidence}")
            
            if confidence < self.face_classifier.similarity_threshold:
                self._set_processing_state(False)
                self.show_snackbar("Face not recognized. Please try again.")
                return

            # Find matching user
            best_match = None
            for user in registered_users:
                stored_email = self.data_cipher.decrypt_data(user['email']).strip().lower()
                if recognized_email.lower() == stored_email:
                    best_match = user
                    break

            if best_match:
                # Process successful login
                fullname = self.data_cipher.decrypt_data(best_match['fullname'])
                email = self.data_cipher.decrypt_data(best_match['email']).strip().lower()
                user_role = best_match['user_role']

                # Store session data
                self.page.client_storage.set("user_data", best_match)
                self.page.client_storage.set("user_role", user_role)
                self.page.client_storage.set("registered_by_admin", False)

                # Special handling for admin
                if user_role == 'Administrator':
                    self.page.client_storage.set("admin_data", best_match)
                    self.show_snackbar(f"Welcome back, Admin {fullname}!")
                    self.show_admin(email=email)
                else:
                    self.show_snackbar(f"Welcome back, {fullname}!")
                    self.show_user(email=email)
            else:
                self._set_processing_state(False)
                self.show_snackbar("Recognized face not in our system. Please register first.")

        except Exception as e:
            self.logger.error(f"Sign-in error: {str(e)}", exc_info=True)
            self._set_processing_state(False)
            self.show_snackbar(f"System error: {str(e)}")

    def show_admin(self, email):
        print("Navigating to User page")
        self.running = False  # Stop the thread
        self.camera_manager.release_camera()
        self._set_processing_state(False)
        update_attendance(email=email, action='sign_in')
        self.page.go('/admin')


    def show_user(self, email):
        print("Navigating to User page")
        self.running = False  # Stop the thread
        self.camera_manager.release_camera()
        self._set_processing_state(False)
        update_attendance(email=email, action='sign_in')
        self.page.go('/user')
