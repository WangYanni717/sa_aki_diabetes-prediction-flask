# ===== Flask Web Deployment Application =====
# Usage: Run in terminal：python app_flask.py
# Access: http://localhost:5001

from flask import Flask, render_template_string, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

app = Flask(__name__)

# Load model
model = None
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    BASE_DIR / "cat_model.pkl",
    BASE_DIR / "2.训练集构建模型" / "cat_model.pkl",
]

for model_path in MODEL_PATHS:
    if model_path.exists():
        try:
            with model_path.open('rb') as f:
                model = pickle.load(f)
            print(f"✅ Model loaded successfully: {model_path}")
            break
        except Exception as e:
            print(f"❌ Model loading failed from {model_path}: {e}")

if model is None:
    print("❌ Model loading failed: no valid model file found")

# Feature list
feature_names = ['uo_ml_kg_24h', 'balance', 'plt_min', 'lac_max', 'weight', 'pt_max', 
                 'glu_max', 'ph_min', 'sofa', 'rdw_max', 'min_ndbp', 'bun_max', 
                 'norepinephrine_rate', 'wbc_max', 'min_hr', 'max_t', 'min_spo2', 'ptt_max', 'age']

# Feature data range constraints (based on training data outlier handling)
feature_ranges = {
    'age': (0, 90),                          # 年龄：0-120岁
    'weight': (30, 200),                      # 体重：30-200 kg
    'uo_ml_kg_24h': (0, 70),                  # 尿输出：0-70 ml/kg/24h
    'balance': (-8000, 10000),                # 液体平衡：-8000 到 10000 ml
    'plt_min': (0, 800),                      # 血小板：≤800 ×10⁹/L
    'lac_max': (0, 20),                       # 乳酸：≤20 mmol/L
    'glu_max': (0, 1000),                     # 血糖：≤1000 mg/dL
    'wbc_max': (0, 100),                      # 白细胞：≤100 ×10⁹/L
    'rdw_max': (0, 30),                       # RDW：≤30 %
    'ph_min': (6.8, 7.8),                     # pH：6.8-7.8
    'norepinephrine_rate': (0, 2),            # 去甲肾上腺素：≤2 μg/kg/min
    'min_ndbp': (0, 120),                     # 舒张压：0-120 mmHg
    'min_spo2': (40, 100),                    # 血氧饱和度：40-100%
    'min_hr': (20, 140),                      # 心率：20-140 bpm
    'max_t': (34, 42),                        # 体温：34-42 ℃
    'pt_max': (0, 100),                       # PT：≤100 s
    'ptt_max': (0, 150),                      # PTT：合理范围
    'bun_max': (0, 240),                      # 尿素氮：合理范围
    'sofa': (0, 24)                           # SOFA评分：0-24分
}

# HTML frontend template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SA_AKI_Diabetes In-Hospital Mortality Prediction Model</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.8em;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.95;
        }
        
        .content {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 30px;
            padding: 50px;
        }
        
        .form-section h2 {
            color: #333;
            margin-bottom: 25px;
            font-size: 1.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .form-group label {
            font-weight: 600;
            color: #333;
            font-size: 0.95em;
        }
        
        .form-group input {
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            transition: all 0.3s;
            background: #f9f9f9;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .btn-predict {
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            width: 100%;
        }
        
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .btn-predict:active {
            transform: translateY(0);
        }
        
        .result-section {
            display: flex;
            flex-direction: column;
            gap: 25px;
            justify-content: flex-start;
        }
        
        .result-section h2 {
            color: #333;
            margin-bottom: 25px;
            font-size: 1.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .result-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        }
        
        .result-card h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .risk-score {
            font-size: 3.5em;
            font-weight: 700;
            color: #667eea;
            text-align: center;
            margin: 20px 0;
        }
        
        .risk-level {
            font-size: 1.3em;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .risk-level.low {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        
        .risk-level.medium {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            border: 2px solid #ffeaa7;
        }
        
        .risk-level.high {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        
        .clinical-advice {
            background: white;
            padding: 20px;
            border-radius: 8px;
            line-height: 1.8;
            color: #555;
            font-size: 0.95em;
            border: 1px solid #e0e0e0;
        }
        
        .risk-indicators {
            background: #f0f4ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .risk-indicators h4 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .indicator-item {
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 5px;
            background: white;
            border-left: 3px solid #ff6b6b;
        }
        
        .indicator-item.warning {
            border-left-color: #ffa94d;
        }
        
        .indicator-item.info {
            border-left-color: #4ecdc4;
        }
        
        .footer {
            background: #f8f9fa;
            padding: 30px 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }
        
        .footer-item {
            padding: 0 20px;
            border-right: 1px solid #e0e0e0;
        }
        
        .footer-item:last-child {
            border-right: none;
        }
        
        .footer-item h4 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .hidden {
            display: none !important;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 1024px) {
            .content {
                grid-template-columns: 1fr;
            }
            
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .footer {
                grid-template-columns: 1fr;
            }
            
            .footer-item {
                border-right: none;
                border-bottom: 1px solid #e0e0e0;
                padding-bottom: 15px;
            }
            
            .footer-item:last-child {
                border-bottom: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 SA_AKI_Diabetes In-Hospital Mortality Prediction Model</h1>
            <p>Auxiliary Clinical Risk Assessment System - Machine Learning-based In-Hospital Mortality Prediction</p>
        </div>
        
        <div class="content">
            <div class="form-section">
                <h2>📝 Patient Information Input</h2>
                <div style="background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 12px 15px; margin-bottom: 20px; color: #004085; font-size: 0.95em;">
                    <strong>📌 Note:</strong> Please enter patient clinical values within 24 hours after ICU admission
                </div>
                <form id="predictionForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Age (years) <span style="color: #999; font-size: 0.9em">[0-90]</span></label>
                            <input type="number" name="age" value="60" min="0" max="90" step="1" required>
                        </div>
                        <div class="form-group">
                            <label>Weight (kg) <span style="color: #999; font-size: 0.9em">[30-200]</span></label>
                            <input type="number" name="weight" value="70" min="30" max="200" step="1" required>
                        </div>
                    </div>
                    
                    <div style="margin-top: 25px; padding-top: 20px; border-top: 2px solid #e0e0e0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 15px;">Urine Output and Fluid Balance</div>
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Urine Output (ml/kg/24h) <span style="color: #999; font-size: 0.9em">[0-70]</span></label>
                                <input type="number" name="uo_ml_kg_24h" value="1" min="0" max="70" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Fluid Balance (ml) <span style="color: #999; font-size: 0.9em">[-8000, 10000]</span></label>
                                <input type="number" name="balance" value="0" min="-8000" max="10000" step="1" required>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 25px; padding-top: 20px; border-top: 2px solid #e0e0e0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 15px;">Hematology Indicators</div>
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Platelet Minimum (×10⁹/L) <span style="color: #999; font-size: 0.9em">[0-800]</span></label>
                                <input type="number" name="plt_min" value="100" min="0" max="800" step="1" required>
                            </div>
                            <div class="form-group">
                                <label>WBC Maximum (×10⁹/L) <span style="color: #999; font-size: 0.9em">[0-100]</span></label>
                                <input type="number" name="wbc_max" value="10" min="0" max="100" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Red Cell Distribution Width (%) <span style="color: #999; font-size: 0.9em">[0-30]</span></label>
                                <input type="number" name="rdw_max" value="13" min="0" max="30" step="0.1" required>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 25px; padding-top: 20px; border-top: 2px solid #e0e0e0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 15px;">Blood Biochemistry</div>
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Lactate Maximum (mmol/L) <span style="color: #999; font-size: 0.9em">[0-20]</span></label>
                                <input type="number" name="lac_max" value="1" min="0" max="20" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Glucose Maximum (mg/dL) <span style="color: #999; font-size: 0.9em">[0-1000]</span></label>
                                <input type="number" name="glu_max" value="150" min="0" max="1000" step="1" required>
                            </div>
                            <div class="form-group">
                                <label>Blood Urea Nitrogen Maximum (mg/dL) <span style="color: #999; font-size: 0.9em">[0-240]</span></label>
                                <input type="number" name="bun_max" value="20" min="0" max="240" step="1" required>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 25px; padding-top: 20px; border-top: 2px solid #e0e0e0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 15px;">Coagulation and Acid-Base</div>
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Prothrombin Time Maximum (s) <span style="color: #999; font-size: 0.9em">[0-100]</span></label>
                                <input type="number" name="pt_max" value="13" min="0" max="100" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Partial Thromboplastin Time Maximum (s) <span style="color: #999; font-size: 0.9em">[0-150]</span></label>
                                <input type="number" name="ptt_max" value="30" min="0" max="150" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>pH Minimum <span style="color: #999; font-size: 0.9em">[6.8-7.8]</span></label>
                                <input type="number" name="ph_min" value="7.35" min="6.8" max="7.8" step="0.01" required>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 25px; padding-top: 20px; border-top: 2px solid #e0e0e0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 15px;">Vital Signs</div>
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Heart Rate Minimum (bpm) <span style="color: #999; font-size: 0.9em">[20-140]</span></label>
                                <input type="number" name="min_hr" value="70" min="20" max="140" step="1" required>
                            </div>
                            <div class="form-group">
                                <label>Temperature Maximum (℃) <span style="color: #999; font-size: 0.9em">[34-42]</span></label>
                                <input type="number" name="max_t" value="37" min="34" max="42" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Oxygen Saturation Minimum (%) <span style="color: #999; font-size: 0.9em">[40-100]</span></label>
                                <input type="number" name="min_spo2" value="95" min="40" max="100" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>Diastolic Blood Pressure Minimum (mmHg) <span style="color: #999; font-size: 0.9em">[0-120]</span></label>
                                <input type="number" name="min_ndbp" value="60" min="0" max="120" step="1" required>
                            </div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 25px; padding-top: 20px; border-top: 2px solid #e0e0e0;">
                        <div style="color: #667eea; font-weight: 600; margin-bottom: 15px;">Medications and Scoring</div>
                        <div class="form-grid">
                            <div class="form-group">
                                <label>Norepinephrine Dose (μg/kg/min) <span style="color: #999; font-size: 0.9em">[0-2]</span></label>
                                <input type="number" name="norepinephrine_rate" value="0" min="0" max="2" step="0.1" required>
                            </div>
                            <div class="form-group">
                                <label>SOFA Score <span style="color: #999; font-size: 0.9em">[0-24]</span></label>
                                <input type="number" name="sofa" value="0" min="0" max="24" step="1" required>
                            </div>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn-predict">🔮 Predict Risk</button>
                </form>
            </div>
            
            <div class="result-section">
                <div id="resultDiv" class="hidden">
                    <h2>📊 Prediction Results</h2>
                    <div class="result-card">
                        <h3>Risk Prediction Probability</h3>
                        <div class="risk-score" id="riskScore">--</div>
                    </div>
                    <div id="riskLevelDiv" class="risk-level"></div>
                    <div class="result-card">
                        <h3>💡 Clinical Recommendations</h3>
                        <div class="clinical-advice" id="clinicalAdvice"></div>
                    </div>
                    <div id="riskIndicatorsDiv" class="result-card hidden">
                        <div class="risk-indicators" id="riskIndicators"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div class="footer-item">
                <h4>📋 Model Information</h4>
                <p>Algorithm: CatBoost<br>Number of Features: 19<br>Purpose: In-Hospital Mortality Prediction</p>
            </div>
            <div class="footer-item">
                <h4>⚠️ Disclaimer</h4>
                <p>This model is for reference only and cannot replace clinical judgment of physicians. Final diagnosis and treatment decisions should be made by healthcare professionals.</p>
            </div>
            <div class="footer-item">
                <h4>🔒 Data Protection</h4>
                <p>Local deployment with no data upload | Patient privacy protected | Compliant with medical data security standards</p>
            </div>
        </div>
    </div>
    
    <script>
        // Add real-time validation
        const form = document.getElementById('predictionForm');
        const inputs = form.querySelectorAll('input[type="number"]');
        
        inputs.forEach(input => {
            input.addEventListener('input', function() {
                const min = parseFloat(this.min);
                const max = parseFloat(this.max);
                const value = parseFloat(this.value);
                const label = this.parentElement.querySelector('label');
                
                // Remove previous error message if exists
                const prevError = this.parentElement.querySelector('.error-message');
                if (prevError) {
                    prevError.remove();
                }
                
                // Show error message if out of range
                if (value < min || value > max) {
                    this.style.borderColor = '#dc3545';
                    this.style.backgroundColor = '#fff5f5';
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'error-message';
                    errorMsg.textContent = `⚠️ Value must be between ${min} and ${max}`;
                    errorMsg.style.color = '#dc3545';
                    errorMsg.style.fontSize = '0.85em';
                    errorMsg.style.marginTop = '4px';
                    errorMsg.style.fontWeight = '500';
                    this.parentElement.appendChild(errorMsg);
                } else {
                    this.style.borderColor = '#e0e0e0';
                    this.style.backgroundColor = '#f9f9f9';
                }
            });
        });
        
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = e.target.querySelector('button');
            const originalText = btn.textContent;
            btn.innerHTML = '<span class="loading"></span>Processing...';
            btn.disabled = true;
            
            const formData = new FormData(e.target);
            const data = {};
            for (let [key, value] of formData.entries()) {
                data[key] = parseFloat(value);
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const riskScore = result.risk_score;
                    document.getElementById('riskScore').textContent = (riskScore * 100).toFixed(1) + '%';
                    
                    let levelClass = 'low';
                    let levelText = '🟢 Low Risk';
                    let advice = '✅ <strong>Predicted adverse event risk is relatively low</strong><br><br>';
                    advice += '• Continue routine monitoring<br>';
                    advice += '• Maintain current treatment plan<br>';
                    advice += '• Regular assessment of patient status';
                    
                    if (riskScore >= 0.7) {
                        levelClass = 'high';
                        levelText = '🔴 High Risk';
                        advice = '❌ <strong>Predicted adverse event risk is high</strong><br><br>';
                        advice += '• <strong>Consider active clinical intervention measures</strong><br>';
                        advice += '• Strengthen critical monitoring (heart rate, oxygen saturation, etc.)<br>';
                        advice += '• Consider escalating treatment plan<br>';
                        advice += '• Notify attending physician and relevant departments<br>';
                        advice += '• Prepare necessary emergency measures';
                    } else if (riskScore >= 0.3) {
                        levelClass = 'medium';
                        levelText = '🟡 Moderate Risk';
                        advice = '⚠️ <strong>Predicted adverse event risk is moderate</strong><br><br>';
                        advice += '• Consider increasing patient monitoring frequency<br>';
                        advice += '• Consider appropriate preventive interventions<br>';
                        advice += '• Closely observe changes in relevant clinical indicators<br>';
                        advice += '• Prepare contingency plans for emergencies';
                    }
                    
                    const riskLevelDiv = document.getElementById('riskLevelDiv');
                    riskLevelDiv.className = 'risk-level ' + levelClass;
                    riskLevelDiv.textContent = levelText;
                    
                    document.getElementById('clinicalAdvice').innerHTML = advice;
                    
                    // Display risk indicators
                    if (result.risk_indicators && result.risk_indicators.length > 0) {
                        let indicatorsHTML = '<h4>🔍 Detected Risk Factors:</h4>';
                        result.risk_indicators.forEach(ind => {
                            indicatorsHTML += '<div class="indicator-item">' + ind + '</div>';
                        });
                        document.getElementById('riskIndicators').innerHTML = indicatorsHTML;
                        document.getElementById('riskIndicatorsDiv').classList.remove('hidden');
                    } else {
                        document.getElementById('riskIndicatorsDiv').classList.add('hidden');
                    }
                    
                    document.getElementById('resultDiv').classList.remove('hidden');
                } else {
                    alert('Prediction failed: ' + result.error);
                }
            } catch (error) {
                alert('Request failed: ' + error.message);
            } finally {
                btn.textContent = originalText;
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded correctly'
            })
        
        data = request.json
        
        # ===== Data Validation =====
        validation_errors = []
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in data:
                value = float(data[feature])
                if value < min_val or value > max_val:
                    validation_errors.append(
                        f"{feature}: value {value} is out of range [{min_val}, {max_val}]"
                    )
        
        if validation_errors:
            return jsonify({
                'success': False,
                'error': 'Input data exceeds reasonable range: ' + '; '.join(validation_errors)
            })
        
        # Create input dataframe
        input_df = pd.DataFrame([data])
        input_df = input_df[feature_names]
        
        # Convert data types
        for col in feature_names:
            input_df[col] = pd.to_numeric(input_df[col])
        
        # Perform prediction
        pred_proba = model.predict_proba(input_df)[0]
        risk_score = float(pred_proba[1])
        
        # Identify risk factors
        risk_indicators = []
        if data.get('sofa', 0) > 6:
            risk_indicators.append("🔴 High SOFA score (>6)")
        if data.get('lac_max', 0) > 2.0:
            risk_indicators.append("🟠 Elevated lactate (>2.0 mmol/L)")
        if data.get('plt_min', 0) < 50:
            risk_indicators.append("🔴 Significant thrombocytopenia (<50×10⁹/L)")
        if data.get('min_spo2', 100) < 90:
            risk_indicators.append("🔴 Low oxygen saturation (<90%)")
        if data.get('ph_min', 7.35) < 7.25:
            risk_indicators.append("🔴 Significant pH decrease (<7.25)")
        if data.get('norepinephrine_rate', 0) > 0.5:
            risk_indicators.append("🟠 High-dose norepinephrine required (>0.5 μg/kg/min)")
        if data.get('pt_max', 0) > 18:
            risk_indicators.append("🟠 Prolonged prothrombin time (>18s)")
        if data.get('bun_max', 0) > 50:
            risk_indicators.append("🟠 Elevated blood urea nitrogen (>50 mg/dL)")
        
        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'no_event_prob': float(pred_proba[0]),
            'risk_indicators': risk_indicators
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/')
def home():
    return "Welcome to the SA_AKI_Diabetes In-Hospital Mortality Prediction Model!"

if __name__ == '__main__':
    # 获取 Render 提供的端口，如果没有设置，默认为 5001
    port = int(os.environ.get('PORT', 5001))

    print("=" * 60)
    print("🏥 SA_AKI_Diabetes In-Hospital Mortality Prediction Model - Flask Server")
    print("=" * 60)
    print("✅ Server starting...")
    print(f"📍 Access URL: http://localhost:{port}")
    print("🛑 Stop service: Press Ctrl+C")
    print("=" * 60)
    
    # 仅在本地开发环境使用，生产环境由Gunicorn启动
    app.run(host='0.0.0.0', port=port, debug=False)
