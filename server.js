const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// 1. Connect to MongoDB
// Replace 'your_mongodb_connection_string' with your actual Compass or Atlas string
mongoose.connect('mongodb://localhost:27017/hospitalDB')
    .then(() => console.log("Connected to MongoDB"))
    .catch(err => console.log("DB Connection Error:", err));

// 2. Define the Patient Schema
const patientSchema = new mongoose.Schema({
    patientID: String,
    name: String,
    doctor: String,
    disease: String,
    date: String,
    status: { type: String, default: 'Visited' }
});

const Patient = mongoose.model('Patient', patientSchema);

// --- ADMIN API ROUTES ---

// GET all records for the dashboard
app.get('/admin/all-records', async (req, res) => {
    try {
        const records = await Patient.find();
        res.json(records);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// EDIT/UPDATE a record
app.put('/admin/update-record/:id', async (req, res) => {
    try {
        const { name, disease } = req.body;
        // findByIdAndUpdate uses the unique MongoDB _id
        await Patient.findByIdAndUpdate(req.params.id, { name, disease });
        res.json({ message: "Record updated successfully" });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// DELETE a record
app.delete('/admin/delete-record/:id', async (req, res) => {
    try {
        await Patient.findByIdAndDelete(req.params.id);
        res.json({ message: "Record deleted successfully" });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.listen(5000, () => console.log("Server running on port 5000"));