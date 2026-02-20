from flask import Flask, render_template, request, send_file, jsonify
import numpy as np
import joblib
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from datetime import datetime
import os

print("App is starting...")

# Initialize Flask app
app = Flask(__name__)

# Try to load model pipeline if available
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_pipeline.pkl')
pipeline = None
model = None
scaler = None
feature_columns = []
if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
        model = pipeline.get('model')
        scaler = pipeline.get('scaler')
        feature_columns = pipeline.get('feature_columns', [])
        print('Loaded fraud_pipeline.pkl')
    except Exception as e:
        print('Failed loading pipeline:', e)
else:
    print('fraud_pipeline.pkl not found; prediction will be disabled')


# ==============================
# Routes
# ==============================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict_page():
    # Render the prediction form page
    return render_template('predict.html')


@app.route('/predict.html', methods=['GET'])
def predict_page_html():
    # Alias to support the landing page's hard link to predict.html
    return render_template('predict.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('index.html')


@app.route('/contact', methods=['GET'])
def contact():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # Handle form POST, compute prediction, and render predict.html with results
    if model is None or scaler is None or not feature_columns:
        # Model not available - show an error message in the same page
        return render_template('predict.html', probability=None, error='Model not available')

    try:
        data = {}
        for col in feature_columns:
            # fall back to 0 if field missing
            val = request.form.get(col, request.form.get(col, 0))
            data[col] = float(val)

        final_features = np.array([[data[col] for col in feature_columns]])
        final_features = scaler.transform(final_features)

        probability = model.predict_proba(final_features)[0][1] * 100
        probability = round(float(probability), 2)

        return render_template('predict.html', probability=probability)

    except Exception as e:
        return render_template('predict.html', probability=None, error=str(e))


@app.route('/investigation', methods=['GET', 'POST'])
def raise_investigation():
    if request.method == 'POST':
        # Accept JSON or form data
        data = {}
        try:
            data = request.get_json() or {}
        except Exception:
            data = {}
        if not data:
            data = request.form.to_dict()

        raised_by = data.get('raised_by', 'Unknown')
        description = data.get('description', '')

        # In a real app we'd persist this to a DB or ticketing system.
        print(f"Investigation raised by: {raised_by}\nDescription: {description}")

        return jsonify({ 'status': 'success', 'message': f'Investigation raised by {raised_by}' })

    # GET fallback: simple message
    return "<h2>🚨 Investigation Case Created Successfully!</h2>"


@app.route('/download_pdf')
def download_report():
    # Read available parameters (front-end supplies them as query params)
    args = request.args
    probability = args.get('probability', 'N/A')
    risk = args.get('risk', 'N/A')
    insights_raw = args.get('insights', '')

    # Known form fields we expect; include any that were submitted
    known_fields = {
        'total_claim_amount': 'Total Claim Amount',
        'injury_claim': 'Injury Claim',
        'property_claim': 'Property Claim',
        'vehicle_claim': 'Vehicle Claim',
        'policy_annual_premium': 'Annual Premium',
        'age': 'Insured Age'
    }

    # Helper formatting
    def parse_prob(pstr):
        try:
            return float(str(pstr).replace('%',''))
        except Exception:
            return None

    def money(v):
        try:
            f = float(v)
            return f"${f:,.2f}"
        except Exception:
            return str(v)

    prob_num = parse_prob(probability)
    # Determine verdict and fraud type heuristics
    verdict = 'N/A'
    if prob_num is None:
        verdict = 'N/A'
    else:
        verdict = 'FRAUDULENT CLAIM' if prob_num >= 70 else 'LEGITIMATE CLAIM'

    fraud_type = 'Pattern Anomaly' if ('pattern' in insights_raw.lower()) else 'N/A'

    # Create PDF
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    p.setFillColorRGB(0.06, 0.75, 0.98)  # cyan-like
    p.setFont('Helvetica-Bold', 22)
    title = 'SmartSecure AI - Fraud Detection Report'
    p.drawCentredString(width/2, height - 50, title)
    p.setFillColorRGB(0,0,0)

    # Timestamp
    p.setFont('Helvetica', 9)
    p.drawString(72, height - 80, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 110

    # Prediction Summary
    p.setFillColorRGB(0.06, 0.75, 0.98)
    p.setFont('Helvetica-Bold', 14)
    p.drawString(72, y, 'Prediction Summary')
    p.setFillColorRGB(0,0,0)
    y -= 20
    p.setFont('Helvetica', 11)
    p.drawString(72, y, verdict)
    y -= 28

    # Risk Assessment Table
    p.setFillColorRGB(0.06, 0.75, 0.98)
    p.setFont('Helvetica-Bold', 14)
    p.drawString(72, y, 'Risk Assessment')
    y -= 18

    table_x = 180
    table_w = 260
    col1_w = 140
    col2_w = table_w - col1_w
    row_h = 22

    # Header row
    p.setFillColor(colors.cyan)
    p.rect(table_x, y - row_h, table_w, row_h, fill=1, stroke=1)
    p.setFillColorRGB(0,0,0)
    p.setFont('Helvetica-Bold', 10)
    p.drawString(table_x + 8, y - 16, 'Metric')
    p.drawString(table_x + col1_w + 8, y - 16, 'Value')
    y -= row_h

    # Rows
    rows = [
        ('Fraud Probability', f"{probability}"),
        ('Risk Level', risk),
        ('Fraud Type', fraud_type)
    ]
    p.setFont('Helvetica', 10)
    for label, val in rows:
        p.setFillColorRGB(0.96,0.96,0.86)
        p.rect(table_x, y - row_h, table_w, row_h, fill=1, stroke=1)
        p.setFillColorRGB(0,0,0)
        p.drawString(table_x + 8, y - 16, label)
        p.drawString(table_x + col1_w + 8, y - 16, str(val))
        y -= row_h

    y -= 18

    # Explainable AI Analysis
    p.setFillColorRGB(0.06, 0.75, 0.98)
    p.setFont('Helvetica-Bold', 14)
    p.drawString(72, y, 'Explainable AI Analysis')
    y -= 18
    p.setFillColorRGB(0,0,0)
    p.setFont('Helvetica', 10)
    if insights_raw:
        insights = insights_raw.split(' || ')
        for ins in insights:
            # bullet
            p.drawString(80, y, u'• ' + ins)
            y -= 14
            if y < 120:
                p.showPage()
                y = height - 72
    else:
        p.drawString(80, y, 'No insights available')
        y -= 14

    y -= 12

    # Claim Details table
    p.setFillColorRGB(0.06, 0.75, 0.98)
    p.setFont('Helvetica-Bold', 14)
    p.drawString(72, y, 'Claim Details')
    y -= 18

    # Table layout
    table_x = 180
    table_w = 260
    col1_w = 160
    col2_w = table_w - col1_w
    row_h = 20

    # Header
    p.setFillColor(colors.cyan)
    p.rect(table_x, y - row_h, table_w, row_h, fill=1, stroke=1)
    p.setFillColorRGB(0,0,0)
    p.setFont('Helvetica-Bold', 10)
    p.drawString(table_x + 8, y - 15, 'Field')
    p.drawString(table_x + col1_w + 8, y - 15, 'Value')
    y -= row_h

    p.setFont('Helvetica', 10)
    for key, label in known_fields.items():
        val = args.get(key, '')
        display = money(val) if key != 'age' else (str(int(float(val))) if val not in (None, '') else '')
        p.setFillColorRGB(0.96,0.96,0.86)
        p.rect(table_x, y - row_h, table_w, row_h, fill=1, stroke=1)
        p.setFillColorRGB(0,0,0)
        p.drawString(table_x + 8, y - 15, label)
        p.drawString(table_x + col1_w + 8, y - 15, display)
        y -= row_h
        if y < 120:
            p.showPage()
            y = height - 72

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="fraud_report.pdf",
        mimetype="application/pdf"
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)