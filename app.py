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

# --- 1. LOAD ENVIRONMENTS ---
load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/healthcare_db")
mongo = PyMongo(app)

# --- CONFIGURATION ---
FAST2SMS_API_KEY = os.getenv('FAST2SMS_API_KEY')

# --- 2. UTILITY FUNCTIONS ---
def send_actual_sms(to_phone, message_body):
    """Utility function to trigger Fast2SMS alerts"""
    try:
        if not FAST2SMS_API_KEY:
            print("SMS Error: Fast2SMS API Key missing in .env")
            return False
            
        clean_phone = str(to_phone).replace("+91", "").strip()
        
        # Change to False for live production
        TEST_MODE = False

        if TEST_MODE:
            print(f"--- TEST MODE: SMS WOULD BE SENT TO {clean_phone} ---")
            print(f"Content: {message_body}")
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
        res_data = response.json()
        
        return res_data.get("return", False)
    except Exception as e:
        print(f"SMS Transmission Failed: {e}")
        return False

def send_email(to_email, subject, body):
    try:
        sender = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
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
            
        return float(clean_val)
    except (ValueError, TypeError):
        return float(default)

# --- 3. AI MODEL LOADING ---
try:
    model = tf.keras.models.load_model("drug_risk_model.h5")
    with open("scaler.pkl", "rb") as f: scaler = pickle.load(f)
    with open("pca.pkl", "rb") as f: pca = pickle.load(f)
    print("AI Model & PCA Loaded Successfully")
except Exception as e:
    print(f"AI Warning: {e}")

# --- 4. BACKGROUND SCHEDULER ---
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

@scheduler.task('cron', id='do_sms_reminders', hour=9, minute=0)
def sms_remainder():
    with app.app_context():
        target = (datetime.now() + timedelta(days=3)).date()
        upcoming = mongo.db.appointments.find({"reminded": False})
        for apt in upcoming:
            if apt['appointmentDate'].date() == target:
                msg = f"Reminder: Appointment with {apt['doctor']} in 3 days."
                if send_actual_sms(apt['phone'], msg):
                    mongo.db.appointments.update_one({"_id": apt["_id"]}, {"$set": {"reminded": True}})

# --- 5. ROUTES ---

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    
    if mongo.db.users.find_one({"username": username}):
        return jsonify({"success": False, "message": "Username exists"}), 400

    hashed_pw = bcrypt.hashpw(data.get('password').encode('utf-8'), bcrypt.gensalt())
    
    new_user = {
        "username": username,
        "password": hashed_pw,
        "role": data.get('role'),
        "phone": data.get('phone'),
        "createdAt": datetime.now()
    }
    mongo.db.users.insert_one(new_user)
    
    if data.get('role') == 'patient':
        current_date = datetime.now().strftime("%Y-%m-%d")
        existing = mongo.db.patient_records.find_one({"name": username})
        patient_id = existing['patientID'] if existing else f"PID-{random.randint(10000, 99999)}"

        mongo.db.patient_records.insert_one({
            "name": username, 
            "patientID": patient_id,
            "phone": data.get('phone'), 
            "date": current_date,
            "lastVisited": current_date,
            "status": "New"
        })
        if data.get('phone'):
            send_actual_sms(data.get('phone'), f"Welcome! Your HealthCare ID is {patient_id}")

    return jsonify({"success": True})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = mongo.db.users.find_one({"username": data['username'], "role": data['role']})
    if user and bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        return jsonify({"success": True, "role": user['role']}), 200
    return jsonify({"success": False}), 401

@app.route('/book-slot', methods=['POST'])
def book_slot():
    data = request.json
    try:
        appt_date = datetime.strptime(data.get('appointmentDate'), '%Y-%m-%d')
    except:
        return jsonify({"success": False, "error": "Invalid date format"}), 400

    booking = {
        "patientID": data.get('patientID'),
        "phone": data.get('phone'),
        "email": data.get('email'),
        "doctor": data.get('doctor'),
        "appointmentDate": appt_date,
        "reminded": False
    }

    mongo.db.appointments.insert_one(booking)

    # Notifications
    phone = data.get('phone')
    if phone:
        msg = f"Appointment booked with {data.get('doctor')} on {data.get('appointmentDate')}"
        send_actual_sms(phone, msg)

    email = data.get('email')
    if email:
        subject = "Appointment Confirmation"
        body = f"Hello,\n\nYour appointment with {data.get('doctor')} on {data.get('appointmentDate')} is confirmed."
        send_email(email, subject, body)

    return jsonify({"success": True})

@app.route('/get-patient-profile/<username>', methods=['GET'])
def get_patient_profile(username):
    record = mongo.db.patient_records.find_one({"name": username}, sort=[("_id", -1)])
    if record:
        return jsonify({
            "isNew": False, 
            "name": record['name'], 
            "patientID": record['patientID'], 
            "currentDate": datetime.now().strftime("%Y-%m-%d"),
            "lastVisited": record.get('lastVisited', "First Time"),
            "age": record.get('age', ''),
            "gender": record.get('gender', '')
        })
    return jsonify({"isNew": True, "name": username, "currentDate": datetime.now().strftime("%Y-%m-%d")})

@app.route('/submit-patient', methods=['POST'])
def submit_patient():
    data = request.json
    pid = data.get('patientID')
    
    existing = mongo.db.patient_records.find_one({"name": data.get('name')})
    if not pid:
        pid = existing['patientID'] if existing else f"PID-{random.randint(10000, 99999)}"
    
    current_date = data.get('date') or datetime.now().strftime("%Y-%m-%d")
    last_visited = existing.get('lastVisited') if existing else current_date

    new_rec = {
        "name": data.get('name'), 
        "patientID": pid, 
        "age": data.get('age'),
        "gender": data.get('gender'), 
        "doctor": data.get('doctor'),
        "disease": data.get('disease'), 
        "medicalData": data.get('medicalData'),
        "date": current_date,
        "lastVisited": last_visited,
        "status": data.get('status', "Waiting"), 
        "riskLevel": "Pending"
    }
    
    mongo.db.patient_records.insert_one(new_rec)
    return jsonify({"success": True, "patientID": pid})

@app.route('/predict-risk', methods=['POST'])
def predict_risk():
    try:
        data = request.json
        pid = data.get('patientID')
        record = mongo.db.patient_records.find_one({"patientID": pid}, sort=[("_id", -1)])
        
        if not record:
            return jsonify({"success": False, "error": "Patient record not found"}), 404

        disease = str(record.get('disease', 'General')).lower()
        m_raw = record.get('medicalData') or {}
        m_norm = {str(k).lower().strip(): v for k, v in m_raw.items()} if isinstance(m_raw, dict) else {}

        def get_val(keys, default_val):
            for k in keys:
                if k in m_norm and m_norm[k] not in [None, ""]:
                    return clean_input(m_norm[k], default_val)
            return default_val

        risk_level = "Low Risk"

        if "diabet" in disease:
            hba1c = get_val(['hba1c', 'hb1ac'], 5.5)
            fbs = get_val(['fbs', 'fasting'], 100.0)
            if hba1c >= 8.5 or fbs >= 250: risk_level = "High Risk"
            elif hba1c >= 7.0 or fbs >= 160: risk_level = "Medium Risk"
        elif any(x in disease for x in ["heart", "cardiac"]):
            echo = get_val(['echo', 'ef'], 55.0)
            if 0 < echo <= 35: risk_level = "High Risk"
            elif 35 < echo <= 45: risk_level = "Medium Risk"
        elif any(x in disease for x in ["hypertension", "bp"]):
            bp = get_val(['bp', 'systolic'], 120)
            if bp >= 180: risk_level = "High Risk"
            elif bp >= 150: risk_level = "Medium Risk"

        mongo.db.patient_records.update_one({"_id": record["_id"]}, {"$set": {"riskLevel": risk_level}})
        return jsonify({"success": True, "risk": risk_level, "patientID": pid})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get-assigned-patients/<doctorName>', methods=['GET'])
def get_assigned_patients(doctorName):
    clean_name = doctorName.replace("Dr.", "").strip()
    today_str = datetime.now().strftime("%Y-%m-%d")
    query = {"doctor": {"$regex": clean_name, "$options": "i"}, "date": today_str}
    patients = list(mongo.db.patient_records.find(query).sort("_id", -1))
    for p in patients: p['_id'] = str(p['_id'])
    return jsonify(patients)

# --- ADMIN ROUTES ---
@app.route('/admin/all-records', methods=['GET'])
def get_all_records():
    doctor = request.args.get('doctor', '').replace("Dr.", "").strip()
    query = {"doctor": {"$regex": doctor, "$options": "i"}} if doctor else {}
    records = list(mongo.db.patient_records.find(query).sort("date", -1))
    for r in records: r['_id'] = str(r['_id']) 
    return jsonify(records)

@app.route('/admin/delete-record/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    mongo.db.patient_records.delete_one({"_id": ObjectId(record_id)})
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(debug=True, port=5000)