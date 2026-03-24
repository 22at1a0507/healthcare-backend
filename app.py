from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_cors import CORS
import bcrypt
import random
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from flask_apscheduler import APScheduler
from bson.objectid import ObjectId
import requests
from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText
import pytz  # Add for timezone handling

# --- 1. LOAD ENVIRONMENTS ---
load_dotenv()

app = Flask(__name__)

# Fix: Configure CORS properly with specific origins for production
CORS(app, resources={r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")}})

# Fix: Add MongoDB connection timeout and SSL settings
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["MONGO_CONNECT_TIMEOUT_MS"] = 5000
app.config["MONGO_SERVER_SELECTION_TIMEOUT_MS"] = 5000
mongo = PyMongo(app)

# --- CONFIGURATION ---
FAST2SMS_API_KEY = os.getenv('FAST2SMS_API_KEY')
# Fix: Add timezone configuration
TIMEZONE = pytz.timezone(os.getenv("TIMEZONE", "Asia/Kolkata"))

@app.route('/')
def home():
    return "✅ Healthcare API is running successfully!"

# --- 2. UTILITY FUNCTIONS ---
def send_actual_sms(to_phone, message_body):
    """Utility function to trigger Fast2SMS alerts"""
    try:
        if not FAST2SMS_API_KEY:
            print("SMS Error: FAST2SMS_API_KEY not configured")
            return False
            
        clean_phone = str(to_phone).replace("+91", "").strip()
        
        # Fix: Make TEST_MODE configurable via environment variable
        TEST_MODE = os.getenv("SMS_TEST_MODE", "False").lower() == "true"

        if TEST_MODE:
            print(f"--- TEST MODE: SMS TO {clean_phone}: {message_body} ---")
            return True 

        url = "https://www.fast2sms.com/dev/bulkV2"
        querystring = {
            "authorization": FAST2SMS_API_KEY,
            "message": message_body,
            "language": "english",
            "route": "q",
            "numbers": clean_phone
        }
        
        headers = {'cache-control': "no-cache"}
        response = requests.request("GET", url, headers=headers, params=querystring)
        
        # Fix: Better error handling for SMS API response
        response_data = response.json()
        if response_data.get("return"):
            return True
        else:
            print(f"SMS API Error: {response_data}")
            return False
    except Exception as e:
        print(f"SMS Error: {e}")
        return False

def send_email(to_email, subject, body):
    try:
        sender = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        
        # Fix: Validate email configuration
        if not sender or not password:
            print("Email Error: EMAIL_USER or EMAIL_PASS not configured")
            return False

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to_email

        # Fix: Add timeout and error handling
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("EMAIL ERROR:", e)
        return False

def clean_input(val, default=0.0):
    """Hardened cleaner: handles units and common formatting"""
    try:
        if val is None or str(val).strip() == "": 
            return float(default)
            clean_val = str(val).lower().replace('%', '').replace('mg/dl', '').replace('g/dl', '').strip()
        if '/' in clean_val: 
            return float(clean_val.split('/')[0])
        # Fix: Handle 'normal' string values
        if clean_val == 'normal':
            return float(default)
        return float(clean_val)
    except (ValueError, TypeError): 
        return float(default)

# --- 3. AI MODEL LOADING ---
# Fix: Add proper error handling and model validation
try:
    model_path = "drug_risk_model.h5"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        print("AI Model loaded successfully")
    else:
        model = None
        print(f"Warning: {model_path} not found")
    
    # Fix: Check if scaler and PCA files exist
    scaler_path = "scaler.pkl"
    pca_path = "pca.pkl"
    
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None
        print(f"Warning: {scaler_path} not found")
        
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
    else:
        pca = None
        print(f"Warning: {pca_path} not found")
        
except Exception as e:
    print(f"AI Model Loading Error: {e}")
    model = None
    scaler = None
    pca = None

# --- 4. BACKGROUND SCHEDULER ---
# Fix: Configure APScheduler properly
class Config:
    SCHEDULER_API_ENABLED = True
    SCHEDULER_TIMEZONE = TIMEZONE

app.config.from_object(Config())

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Fix: Fix the cron job function name and logic
@scheduler.task('cron', id='do_sms_reminders', hour=9, minute=0)
def sms_reminder_task():
    """Send SMS reminders for appointments in 3 days"""
    with app.app_context():
        try:
            # Fix: Use timezone-aware datetime
            target_date = (datetime.now(TIMEZONE) + timedelta(days=3)).date()
            
            # Fix: Query appointments that haven't been reminded yet
            upcoming = mongo.db.appointments.find({"reminded": {"$ne": True}})
            
            for apt in upcoming:
                # Fix: Handle both string and datetime appointmentDate
                apt_date = apt['appointmentDate']
                if isinstance(apt_date, str):
                    apt_date = datetime.fromisoformat(apt_date).date()
                elif isinstance(apt_date, datetime):
                    apt_date = apt_date.date()
                
                if apt_date == target_date:
                    msg = f"Reminder: Appointment with {apt['doctor']} in 3 days."
                    if send_actual_sms(apt.get('phone'), msg):
                        mongo.db.appointments.update_one(
                            {"_id": apt["_id"]}, 
                            {"$set": {"reminded": True}}
                        )
        except Exception as e:
            print(f"Scheduler Error: {e}")

# --- 5. ROUTES ---

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
            
        username = data.get('username')
        password = data.get('password')
        role = data.get('role')
        phone = data.get('phone')
        
        # Fix: Validate required fields
        if not username or not password or not role:
            return jsonify({"success": False, "message": "Missing required fields"}), 400
        
        # Fix: Check if username exists
        if mongo.db.users.find_one({"username": username}):
            return jsonify({"success": False, "message": "Username exists"}), 400

        # Fix: Hash password properly
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Safe lookup for existing patient ID
        existing = mongo.db.patient_records.find_one({"name": username})
        if existing and 'patientID' in existing:
            patient_id = existing['patientID']
        else:
            patient_id = f"PID-{random.randint(10000, 99999)}"

        new_user = {
            "username": username,
            "password": hashed_pw,
            "role": role,
            "phone": phone,
            "patientID": patient_id,
            "createdAt": datetime.now(TIMEZONE)
        }
        mongo.db.users.insert_one(new_user)
        
        if role == 'patient':
            current_date = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
            mongo.db.patient_records.insert_one({
                "name": username, 
                "patientID": patient_id,
                "phone": phone, 
                "date": current_date,
                "lastVisited": current_date,
                "status": "New"
            })
            if phone:
                send_actual_sms(phone, f"Welcome! Your HealthCare ID is {patient_id}")

        return jsonify({"success": True, "patientID": patient_id})
    except Exception as e:
        print(f"Registration Error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
            
        username = data.get('username')
        password = data.get('password')
        role = data.get('role')
        
        if not username or not password or not role:
            return jsonify({"success": False, "message": "Missing required fields"}), 400
        
        user = mongo.db.users.find_one({"username": username, "role": role})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return jsonify({
                "success": True, 
                "role": user['role'],
                "username": user['username'],
                "patientID": user.get('patientID', '')
            }), 200
        return jsonify({"success": False, "message": "Invalid credentials"}), 401
    except Exception as e:
        print(f"Login Error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/book-slot', methods=['POST'])
def book_slot():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        try:
            appt_date = datetime.strptime(data.get('appointmentDate'), '%Y-%m-%d')
            appt_date = TIMEZONE.localize(appt_date)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}), 400

        # Fix: Validate required fields
        required_fields = ['patientID', 'phone', 'doctor']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400

        booking = {
            "patientID": data.get('patientID'),
            "phone": data.get('phone'),
            "email": data.get('email'),
            "doctor": data.get('doctor'),
            "appointmentDate": appt_date,
            "reminded": False,
            "createdAt": datetime.now(TIMEZONE)
        }
        mongo.db.appointments.insert_one(booking)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Book Slot Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get-patient-profile/<username>', methods=['GET'])
def get_patient_profile(username):
    try:
        # Fix: Use find_one with sort and handle missing records
        record = mongo.db.patient_records.find_one({"name": username}, sort=[("_id", -1)])
        if record:
            return jsonify({
                "isNew": False, 
                "name": record.get('name'), 
                "patientID": record.get('patientID'), 
                "currentDate": datetime.now(TIMEZONE).strftime("%Y-%m-%d"),
                "lastVisited": record.get('lastVisited', "First Time")
            })
        return jsonify({"isNew": True, "name": username, "currentDate": datetime.now(TIMEZONE).strftime("%Y-%m-%d")})
    except Exception as e:
        print(f"Get Patient Profile Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit-patient', methods=['POST'])
def submit_patient():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        name = data.get('name')
        if not name:
            return jsonify({"success": False, "error": "Name is required"}), 400
            
        existing = mongo.db.patient_records.find_one({"name": name})
        
        if data.get('patientID'):
            pid = data.get('patientID')
        elif existing and 'patientID' in existing:
            pid = existing['patientID']
        else:
            pid = f"PID-{random.randint(10000, 99999)}"
        
        current_date = data.get('date') or datetime.now(TIMEZONE).strftime("%Y-%m-%d")
        
        new_rec = {
            "name": name, 
            "patientID": pid, 
            "age": data.get('age'),
            "gender": data.get('gender'), 
            "doctor": data.get('doctor'),
            "disease": data.get('disease'), 
            "medicalData": data.get('medicalData'),
            "date": current_date,
            "status": data.get('status', "Waiting"), 
            "riskLevel": "Pending",
            "createdAt": datetime.now(TIMEZONE)
        }
        mongo.db.patient_records.insert_one(new_rec)
        return jsonify({"success": True, "patientID": pid})
    except Exception as e:
        print(f"Submit Patient Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/predict-risk', methods=['POST'])
def predict_risk():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        pid = data.get('patientID')
        if not pid:
            return jsonify({"success": False, "error": "Patient ID is required"}), 400
            
        record = mongo.db.patient_records.find_one({"patientID": pid}, sort=[("_id", -1)])
        
        if not record:
            return jsonify({"success": False, "error": "Record not found"}), 404

            disease = str(record.get('disease', 'General')).lower()
            m_raw = record.get('medicalData') or {}
            m_norm = {str(k).lower().strip(): v for k, v in m_raw.items()} if isinstance(m_raw, dict) else {}

            risk_level = "Low Risk"
        
        # Simple Logic Example
        if "diabet" in disease:
            hba1c = clean_input(m_norm.get('hba1c'), 5.5)
            if hba1c >= 7.0:
                risk_level = "High Risk"
            elif hba1c >= 6.5:
                risk_level = "Moderate Risk"
        
        # Fix: Update the record with risk level
        mongo.db.patient_records.update_one(
            {"_id": record["_id"]}, 
            {"$set": {"riskLevel": risk_level}}
        )
        return jsonify({"success": True, "risk": risk_level})
    except Exception as e:
        print(f"Predict Risk Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get-assigned-patients/<doctorName>', methods=['GET'])
def get_assigned_patients(doctorName):
    try:
        clean_name = doctorName.replace("Dr.", "").strip()
        today_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
        query = {"doctor": {"$regex": clean_name, "$options": "i"}, "date": today_str}
        patients = list(mongo.db.patient_records.find(query).sort("_id", -1))
        
        # Fix: Convert ObjectId to string for JSON serialization
        for p in patients:
            p['_id'] = str(p['_id'])
            if 'createdAt' in p and isinstance(p['createdAt'], datetime):
                p['createdAt'] = p['createdAt'].isoformat()
            if 'appointmentDate' in p and isinstance(p.get('appointmentDate'), datetime):
                p['appointmentDate'] = p['appointmentDate'].isoformat()
        
        return jsonify(patients)
    except Exception as e:
        print(f"Get Assigned Patients Error: {e}")
        return jsonify({"error": str(e)}), 500

# Fix: Add error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Fix: Set debug mode based on environment
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
