import flet as ft
import numpy as np
import cvzone, logging
import os, cv2, json, base64, threading, time
from datetime import datetime as dt
from pages.utils import FaceDetector, FaceClassifier, center_crop_frame, update_attendance, DataCipher

class RegisterFace(ft.UserControl):
    def __init__(self, page, camera_manager):
        super().__init__()
        self.page = page
        self.running = True
        self.face_detector = FaceDetector()
        self.data_cipher = DataCipher()
        self.face_classifier = FaceClassifier()
        self.camera_manager = camera_manager
        self.camera = self.camera_manager.get_camera()
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

        self.capture_face_button = ft.Row(
            controls=[
                ft.Container(
                    border_radius=5,
                    expand=True,
                    bgcolor='#3b82f6',
                    gradient=ft.LinearGradient(
                        colors=['#bbf7d0', '#86efac', '#3b82f6'],
                    ),
                    content=ft.Text(
                        'CAPTURE FACE AND GRANT ACCESS',
                        text_align=ft.TextAlign.CENTER,
                        size=20
                    ),
                    padding=ft.padding.only(left=170, right=170, top=10, bottom=10),
                    on_click=self.capture_image
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
        #self.running = True # This was the fix that caused the system not to lag again... Thank God for the help.
        self.update_cam_timer()

    def will_unmount(self):
        self.running = False
        if self.camera is not None:
            self.camera_manager.release_camera()

    def update_cam_timer(self):
        # start a thread to update the camera feed
        def update():
            while self.running:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    if not self.running:  # Stop if the class is unmounted or navigation occurred
                        break
                    self.show_snackbar('Error: Failed to grab frame')
                    #time.sleep(0.1)
                    continue

                try:
                    # detect face and draw bounding box
                    face_loc = self.face_detector.detect_face(frame)
                    if face_loc:
                        x, y, width, height = face_loc
                        cvzone.cornerRect(frame, (x, y, width, height), l=30, t=5, colorR=(0, 255, 0))
                    # get center crop coordinates
                    cropped_frame = center_crop_frame(frame)
                    # convert to base64
                    _, img_arr = cv2.imencode('.png', cropped_frame)
                    img_b64 = base64.b64encode(img_arr)
                    # update image in UI
                    self.img.src_base64 = img_b64.decode('utf-8')
                    self.update()
                except Exception as e:
                    self.show_snackbar(f'Error processing frame: {e}')
                time.sleep(0.033) # cap at ~ 30 FPS
        threading.Thread(target=update, daemon=True).start()

    def build(self):
        return ft.Column(
            [
                ft.Divider(height=10, color='transparent'),
                ft.Text('Position face for capturing', size=24, weight=ft.FontWeight.BOLD, text_align='center'),
                ft.Divider(height=20, color='transparent'),
                self.img,
                self.progress_status,
                self.prog_bar,
                ft.Divider(height=20, color='transparent'),
                self.capture_face_button
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

    def show_snackbar(self, message):
        snackbar = ft.SnackBar(
            bgcolor=ft.colors.GREY_900,
            content=ft.Text(message, color=ft.colors.WHITE)
        )
        self.page.overlay.append(snackbar)
        snackbar.open = True
        self.page.update()

    def capture_image(self, e=None) -> None:
        """Handle face capture and registration process."""
        try:
            # Show processing state
            self._set_processing_state(True, "Processing face...")

            # Validate camera
            camera = self.camera_manager.get_camera()
            if not camera or not camera.isOpened():
                self._set_processing_state(False)
                self.show_snackbar("Camera not available")
                return

            # Capture frame
            ret, frame = camera.read()
            if not ret:
                self._set_processing_state(False)
                self.show_snackbar("Failed to capture image")
                return

            # Validate user data
            user_data = {
                'fullname': self.page.client_storage.get("fullname"),
                'email': self.page.client_storage.get("email"),
                'telephone': self.page.client_storage.get("telephone"),
                'user_role': self.page.client_storage.get("user_role")
            }
            
            if None in user_data.values():
                self._set_processing_state(False)
                self.show_snackbar("Missing registration data")
                return

            # Process face registration in a thread
            def process_registration():
                try:
                    self._set_processing_state(True, "Generating face embeddings...")
                    success = self._register_user(frame, user_data)
                    if success:
                        self._set_processing_state(True, "Finalizing registration...")
                        self._complete_registration(user_data)
                finally:
                    self._set_processing_state(False)

            # Start processing in background thread
            threading.Thread(target=process_registration, daemon=True).start()
                
        except Exception as e:
            self.logger.error(f"Registration failed: {str(e)}", exc_info=True)
            self._set_processing_state(False)
            self.show_snackbar("Registration error occurred", is_error=True)

    def _register_user(self, frame: np.ndarray, user_data: dict) -> bool:
        """Handle face processing and embedding generation."""
        try:
            # Save original image
            faces_dir = os.path.join('assets', 'application_data', 'user_storage', user_data['email'])
            os.makedirs(faces_dir, exist_ok=True)
            image_path = os.path.join(faces_dir, f"{user_data['email']}.jpg")
            cv2.imwrite(image_path, frame)
            
            # Register face embeddings
            self._set_processing_state(True, "Generating face embeddings...")
            if not self.face_classifier.register_user(email=user_data['email'], image_path=image_path):
                raise ValueError("Face registration failed")
            
            # Train classifier (optional - can be done separately)
            self._set_processing_state(True, "Training classifier...")
            if not self.face_classifier.train_classifier():
                self.logger.warning("Classifier training completed with warnings")

            return True
            
        except Exception as e:
            self.logger.error(f"Face processing failed: {str(e)}")
            self.show_snackbar(f"Face processing error: {str(e)}")
            return False

    def _complete_registration(self, user_data: dict) -> None:
        """Finalize user registration and navigate to appropriate page."""
        try:
            # Encrypt sensitive data
            encrypted_data = {
                "fullname": self.data_cipher.encrypt_data(user_data['fullname']),
                "email": self.data_cipher.encrypt_data(user_data['email']),
                "telephone": self.data_cipher.encrypt_data(user_data['telephone']),
                "user_role": user_data['user_role'],
                "face_image": f"assets/application_data/user_storage/{user_data['email']}/{user_data['email']}.jpg",
                "total_attendance": 0,
                "attendance_status": [],
                "last_attendance_time": dt.now().strftime("%d-%m-%Y %H:%M:%S")
            }

            # Update user database
            data_file = os.path.join('assets', 'application_data', 'application_storage', 'registered_data.json')
            users = []
            if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
                with open(data_file, 'r') as f:
                    users = json.load(f)
            users.append(encrypted_data)
            
            with open(data_file, 'w') as f:
                json.dump(users, f, indent=4)

            # Store session data
            self.page.client_storage.set("user_data", encrypted_data)
            self.page.client_storage.set("registered_by_admin", True)
            
            # Navigate based on role
            email = user_data['email'].strip().lower()
            update_attendance(email=email, action='sign_in')
            
            if user_data['user_role'] == 'Administrator':
                self.page.client_storage.set("admin_data", encrypted_data)
                self._navigate_to_admin(email)
            else:
                self._navigate_to_user(email)
                
        except Exception as e:
            self.logger.error(f"Registration completion failed: {str(e)}", exc_info=True)
            raise

    def _navigate_to_admin(self, email: str) -> None:
        """Handle admin navigation cleanup."""
        self.running = False
        self.camera_manager.release_camera()
        self._set_processing_state(False)
        self.show_snackbar('Admin registered successfully!')
        self.page.go('/admin')

    def _navigate_to_user(self, email: str) -> None:
        """Handle user navigation cleanup."""
        self.running = False
        self.camera_manager.release_camera()
        self._set_processing_state(False)
        self.show_snackbar('User registered successfully!')
        self.page.go('/user')
