---

# **Face Recognition Attendance System**  

A secure, scalable attendance system using **real-time facial recognition**, designed for both single-user and multi-user environments. Built with Flet, MTCNN, and TensorFlow.  

---

## **✨ Key Features**  
1. **Dual-Mode Recognition**  
   - **Multi-user mode:** SVM classifier for ≥2 registered users.  
   - **Single-user mode:** OneClassSVM for outlier detection (ideal for admin-only setups).  

2. **Enhanced Security**  
   - Encrypted user data (AES-256 via `cryptography`).  
   - Normalized confidence scores (`[0, 1]`) for consistent thresholds.  

3. **Real-Time Processing**  
   - Live camera feed with face alignment and augmentation.  
   - Adaptive similarity thresholds (configurable per deployment).  

4. **Attendance Management**  
   - Automated sign-in/sign-out logs with 24-hour cooldown.  
   - CSV export for admins.  

5. **Admin Privileges**  
   - Exclusive user registration rights.  
   - Access to raw attendance logs.  

---

## **🛠 Technologies Used**  
| Component           | Technology Stack |  
|---------------------|------------------|  
| **Frontend**        | Flet (Python)    |  
| **Face Detection**  | MTCNN            |  
| **Face Recognition**| TensorFlow + SVM/OneClassSVM |  
| **Data Encryption** | Fernet (AES-256) |  
| **Storage**         | JSON + CSV       |  

---

## **🚀 Usage**  
### **1. Registration**  
```bash
python register_face.py  # Captures face + encrypts user data
```  
- **Admin role required** for new registrations.  

### **2. Attendance**  
```bash
python signin.py  # Real-time sign-in/sign-out  
```  
- **Thresholds**: Adjust `similarity_threshold` in `signin.py` (default: `0.5`).  

### **3. Admin Controls**  
- Export logs:  
  ```python
  from utils import export_to_csv
  export_to_csv("attendance_logs.csv")
  ```  

---

## **🔧 Technical Improvements**  
- **Dual SVM Support**: Seamless fallback between `SVC` (multi-user) and `OneClassSVM` (single-user).  
- **Normalized Scores**: Decision scores scaled to `[0, 1]` for unified thresholds.  
- **Thread Safety**: Camera resources released on app exit.  

---

## **📊 Example Output**  
```plaintext
[SYSTEM] Recognized: Known (0.82)
[ATTENDANCE] User: admin@org.com, Action: sign_in, Time: 2024-03-15 09:00:00  
```  

---

## **🤝 Contributors**  
**Igwe Ugochukwu Edwin**  
- GitHub: [@yourhandle](https://github.com/yourhandle)  
- Email: your.email@example.com  

**Open to collaborations!** Fork → Improve → PR.  

--- 

### **📌 Notes**  
- For production, set `min_sign_in_interval = 86400` (24 hours) in `utils.py`.  
- Tested on Python 3.7.6 (Ubuntu/Windows).  

---

This version highlights technical depth while maintaining clarity. Let me know if you'd like to emphasize any other aspects!
