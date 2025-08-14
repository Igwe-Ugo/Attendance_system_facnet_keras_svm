import pickle
import logging
import numpy as np
import cv2, os, json
import tensorflow as tf
from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN
from datetime import datetime as dt
from keras.models import load_model
from cryptography.fernet import Fernet
from typing import Optional, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("error_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceClassifier:
    def __init__(self):
        self.img_size = (160, 160)
        self.facenet_path = 'assets/application_data/application_storage/facenet_keras.h5'
        self.embedding_file = 'assets/application_data/application_storage/multiple_embeddings.npz'
        self.classifier_file = 'assets/application_data/application_storage/svm_classifier.pkl'
        self.user_dir = 'assets/application_data/user_storage'
        self.similarity_threshold = 0.6  # Minimum cosine similarity for recognition

        self.facenet_model = load_model(self.facenet_path)
        self.face_detector = MTCNN()
        self.svm = SVC(kernel='linear', probability=True)
        self.encoder = LabelEncoder()

        self._ensure_dirs()
        self._load_classifier()
        self.mean_embeddings = self._load_mean_embeddings()

    def _ensure_dirs(self):
        os.makedirs(self.user_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.classifier_file), exist_ok=True)

    def _load_classifier(self):
        if os.path.exists(self.classifier_file):
            try:
                with open(self.classifier_file, 'rb') as f:
                    model = pickle.load(f)
                    if isinstance(model, SVC):
                        self.svm = model
                        logger.info("Loaded multi-class SVM")
            except Exception as e:
                logger.warning(f"Classifier load failed: {e}")

    def _load_mean_embeddings(self) -> Dict[str, np.ndarray]:
        """Load mean embeddings from the structured user storage directory."""
        mean_embeddings = {}
        user_storage_dir = os.path.join('assets', 'application_data', 'user_storage')
        
        if not os.path.exists(user_storage_dir):
            os.makedirs(user_storage_dir, exist_ok=True)
            return mean_embeddings

        for user_email in os.listdir(user_storage_dir):
            user_dir = os.path.join(user_storage_dir, user_email)
            if os.path.isdir(user_dir):
                embedding_path = os.path.join(user_dir, 'mean_embeddings.npy')
                if os.path.exists(embedding_path):
                    try:
                        mean_embeddings[user_email] = np.load(embedding_path)
                        logger.info(f"Loaded embeddings for user: {user_email}")
                    except Exception as e:
                        logger.error(f"Failed to load embeddings for {user_email}: {str(e)}")
        
        return mean_embeddings

    def augment_image(self, image_tensor):
        """Generate augmented versions of the input image"""
        return {
            "original": tf.image.resize(image_tensor, self.img_size),
            "flip": tf.image.flip_left_right(image_tensor),
            "bright+": tf.image.adjust_brightness(image_tensor, 0.2),
            "bright-": tf.image.adjust_brightness(image_tensor, -0.2),
            "contrast+": tf.image.adjust_contrast(image_tensor, 2.0),
            "contrast-": tf.image.adjust_contrast(image_tensor, 0.5)
        }

    def extract_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and extract face using MTCNN"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detector.detect_faces(rgb_image)
            if results:
                x, y, w, h = results[0]['box']
                face = image[y:y+h, x:x+w]
                return cv2.resize(face, self.img_size).astype('float32') / 255.0
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
        return None

    def get_embedding(self, face: np.ndarray) -> Optional[np.ndarray]:
        """Generate FaceNet embedding"""
        try:
            mean, std = face.mean(), face.std()
            face = (face - mean) / std
            emb = self.facenet_model.predict(np.expand_dims(face, axis=0))[0]
            return emb / np.linalg.norm(emb)  # Normalize
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def register_user(self, email: str, image_path: str) -> bool:
        """Register a new user with face augmentation (optimized version)"""
        try:
            # 1. Load and validate original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Invalid image path")

            # 2. Generate and save augmented images (without processing)
            augmented_images = []
            rgb_tensor = tf.convert_to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=tf.float32)
            
            for aug_name, aug_tensor in self.augment_image(rgb_tensor).items():
                aug_img = tf.clip_by_value(aug_tensor, 0, 255).numpy().astype("uint8")
                aug_path = os.path.join(self.user_dir, email, "augmented_images", f"{aug_name}.jpg")
                os.makedirs(os.path.dirname(aug_path), exist_ok=True)
                cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                augmented_images.append(aug_path)
            print(augmented_images)
            
            # 3. Process all saved images in batch
            embeddings = []
            for img_path in augmented_images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                face = self.extract_faces(img)
                if face is not None:
                    emb = self.get_embedding(face)
                    if emb is not None:
                        embeddings.append(emb)

            if not embeddings:
                raise ValueError("No valid faces detected")

            # 4. Save embeddings
            mean_embedding = np.mean(embeddings, axis=0)
            mean_emb_path = os.path.join(self.user_dir, email, 'mean_embedding.npy')
            np.save(mean_emb_path, mean_embedding)
            
            # Update global embeddings file
            if os.path.exists(self.embedding_file):
                data = np.load(self.embedding_file, allow_pickle=True)
                x, y = list(data['embeddings']), list(data['labels'])
            else:
                x, y = [], []

            x.extend(embeddings)
            y.extend([email] * len(embeddings))
            print(y)
            np.savez_compressed(self.embedding_file, embeddings=x, labels=y)

            return True
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False

    def train_classifier(self) -> bool:
        """Train SVM classifier"""
        try:
            if not os.path.exists(self.embedding_file):
                logger.warning("No embeddings found for training")
                return False

            data = np.load(self.embedding_file, allow_pickle=True)
            x, y = np.array(data['embeddings']), np.array(data['labels'])
            self.encoder.fit(y)
            y = self.encoder.transform(y)
            unique_users = np.unique(y)

            if len(unique_users) >= 2:
                self.svm.fit(x, y)
                with open(self.classifier_file, 'wb') as f:
                    pickle.dump(self.svm, f)
                logger.info(f"Trained SVM for {len(unique_users)} users")
                return True
            else:
                logger.warning("Not enough users for training")
                return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    def recognize_face(self, image: np.ndarray) -> Tuple[str, float]:
        """Recognize face using cosine similarity or SVM based on user count"""
        try:
            face = self.extract_faces(image)
            if face is None:
                return "Unknown", 0.0

            emb = self.get_embedding(face)
            if emb is None:
                return "Unknown", 0.0
            
            if len(self.svm.classes_) >= 2:
                if os.path.exists(self.classifier_file):
                    with open(self.classifier_file, 'rb') as f:
                        model = pickle.load(f)

                    if isinstance(model, SVC):
                        face_name = model.predict([emb])[0]
                        probs = model.predict_proba([emb])[0]
                        actual_name = self.encoder.inverse_transform(face_name)
                        max_idx = np.argmax(probs)
                        return actual_name, probs[max_idx]

                return "Unknown, path not found!", 0.0
            
            elif self.mean_embeddings:
                best_match = None
                highest_similarity = 0.0
                
                # Compare with each registered user's mean embedding
                for email, mean_emb in self.mean_embeddings.items():
                    similarity = cosine_similarity([emb], [mean_emb])[0][0]
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = email
                
                if highest_similarity >= self.similarity_threshold:
                    return best_match, highest_similarity
                else:
                    return "Unknown", 0.0

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return "Unknown", 0.0

class FaceDetector:
    def __init__(self):
        """Initialize MTCNN face detector."""
        self.detector = MTCNN()

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect faces in an image and return bounding box."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(rgb_image)
            
            if not results:
                return None
                
            # Get the largest face
            largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = largest_face['box']
            
            # Add padding
            padding = 20
            height, width = image.shape[:2]
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            
            return (x, y, w, h)
        except Exception as e:
            print(f"Face detection failed: {str(e)}")
            return None


class CameraManager:
    '''
    Responsible for all camera functions and inputs...
    '''
    def __init__(self):
        self.camera = None

    def get_camera(self):
        if self.camera is None or not self.camera.isOpened():
            for index in range(10): # Try different camera indices
                self.camera = cv2.VideoCapture(index)
                if self.camera.isOpened():
                    # set camera resolution
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    return self.camera
        return self.camera
    
    def release_camera(self):
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None

# To crop and center the camera frame
def center_crop_frame(frame, size=400):
    height, width = frame.shape[:2]
    start_x = max(0, (width - size) // 2)
    start_y = max(0, (height - size) // 2)
    cropped = frame[start_y:start_y + size, start_x:start_x + size]
    # ensure correct size
    if cropped.shape[:2] != (size, size):
        cropped = cv2.resize(cropped, (size, size))
    return cropped

def update_attendance(email, action):
    '''
        Function responsible for taking attendance in this system.
    '''
    file_path = 'assets/application_data/application_storage/registered_data.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            all_users = json.load(f)
    else:
        print("Error: No registered users found.")
        return

    # Find the user by email
    user_found = False
    for user in all_users:
        # Decrypt and normalize the stored email
        stored_email = DataCipher().decrypt_data(user['email']).strip().lower()
        input_email = email.strip().lower()  # Normalize input email
        print(f"Encrypted Email in JSON: {user['email']}")
        print(f"Decrypted Email for Comparison: '{stored_email}'")
        print(f"Input Email for Comparison: '{input_email}'")
        if stored_email == input_email:
            user_found = True
            if 'attendance_status' not in user:
                user['attendance_status'] = []

            if action == "sign_in":
                # Check if the user can sign in based on the last sign-out time
                if user['attendance_status']:
                    last_record = user['attendance_status'][-1]
                    if last_record['sign_out_time']:
                        date_time_object = dt.strptime(last_record['sign_out_time'], "%d-%m-%Y %H:%M:%S")
                        seconds_elapsed = (dt.now() - date_time_object).total_seconds()
                        min_sign_in_interval = 30  # Replace 30 with 86400 for 24 hours
                        if seconds_elapsed < min_sign_in_interval:
                            print(f"User {email} signed out at {last_record['sign_out_time']}. You can only sign in again after 24 hours.")
                            return  # Stop further processing
                # Append a new sign-in record
                attendance = {
                    'sign_in_time': dt.now().strftime("%d-%m-%Y %H:%M:%S"),
                    'sign_out_time': ''  # Initially empty
                }
                user['total_attendance'] += 1
                user['attendance_status'].append(attendance)
                print(f"Sign-in time recorded for {email}.")

            elif action == "sign_out":
                # Update the last sign-out time
                if user['attendance_status']:
                    last_record = user['attendance_status'][-1]
                    if last_record['sign_out_time'] == '':
                        last_record['sign_out_time'] = dt.now().strftime("%d-%m-%Y %H:%M:%S")
                        user['last_attendance_time'] = dt.now().strftime("%d-%m-%Y %H:%M:%S")
                        print(f"Sign-out time recorded for {email}.")
                    else:
                        print(f"Error: User {email} has already signed out. Sign in first.")
                else:
                    print(f"Error: No sign-in record found for {email}. Please sign in first.")
            else:
                print(f"Error: Invalid action '{action}'. Use 'sign_in' or 'sign_out'.")

            break

    if not user_found:
        print(f"Error: No user found with email {email}. Please register first.")
    else:
        # Save updated data
        with open(file_path, 'w') as f:
            json.dump(all_users, f, indent=4)

class DataCipher:
    def __init__(self, key_file='encryption_key.key'):
        self.key_file = key_file
        self.cipher = self._load_or_generate_key()

    def _load_or_generate_key(self):
        '''
        Load the encryption key from a file or generate a new one
        '''
        try:
            # try to load the key from the key file
            save_key = os.path.join('assets', 'application_data')
            save_key = os.path.join(save_key, 'application_storage')
            os.makedirs(save_key, exist_ok=True)
            register_key = os.path.join(save_key, self.key_file)
            with open(register_key, 'rb') as key_file:
                key = key_file.read()
        except FileNotFoundError:
            # generate a new key if the file doesn't exist
            key = Fernet.generate_key()
            with open(register_key, 'wb') as key_file:
                key_file.write(key)
        return Fernet(key)
    
    def encrypt_data(self, data: str) -> str:
        '''Encrypt a string'''
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, data: str) -> str:
        '''Decrypt a string'''
        return self.cipher.decrypt(data.encode()).decode()
    
