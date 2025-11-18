# File: app.py
from flask import Flask, request, render_template, jsonify
import joblib
import re

app = Flask(__name__)

# Load the trained model + vectorizer saved earlier
MODEL_PATH = "model/email_tone_model.pkl"
VECT_PATH = "model/tfidf_vectorizer.pkl"
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

# Simple rule set (same as your local predict.py) - extendable
RULES = [
    (r'\bdear\b', 'formal'),
    (r'\bsir\b', 'formal'),
    (r'\bmadam\b', 'formal'),
    (r'\bplease respond\b', 'urgent'),
    (r'\brespond immediately\b', 'urgent'),
    (r'\basap\b', 'urgent'),
    (r'\burgent\b', 'urgent'),
    (r'\bkindly\b', 'polite'),
    (r'\bcould you please\b', 'polite'),
    (r'\bplease\b', 'polite'),
    (r'\bthank you\b', 'appreciative'),
    (r'\bthanks\b', 'appreciative'),
    (r'\bi appreciate\b', 'appreciative'),
    (r'\bwhy\b', 'rude'),
    (r'\bunacceptable\b', 'rude'),
    (r'\bfix it\b', 'rude'),
]


def apply_rules(text):
    t = text.lower()
    for patt, tone in RULES:
        if re.search(patt, t):
            return tone
    return None


@app.route("/", methods=["GET"]) 
def index():
    # Render a simple UI
    return render_template("index.html")


@app.route("/predict", methods=["POST"]) 
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data['text']
    text = (text or '').strip()
    if text == "":
        return jsonify({"tone": None, "error": "Empty input"}), 200

    # 1) rules
    rule_tone = apply_rules(text)
    if rule_tone:
        return jsonify({"tone": rule_tone, "method": "rule"}), 200

    # 2) model
    X = vectorizer.transform([text])
    try:
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        best_idx = probs.argmax()
        pred = classes[best_idx]
        confidence = float(probs[best_idx])
    except Exception:
        pred = model.predict(X)[0]
        confidence = None

    return jsonify({"tone": str(pred), "confidence": confidence, "method": "model"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



# -----------------------------------------------------------------
# File: templates/index.html
# Place this file in a folder named "templates" (Flask will auto-find it)

"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Email Tone Classifier</title>
    <style>
      body { font-family: Arial, sans-serif; max-width:800px; margin:40px auto; }
      textarea { width:100%; height:140px; font-size:14px; padding:10px; }
      button { padding:10px 18px; font-size:16px; margin-top:10px; }
      .result { margin-top:20px; padding:12px; border-radius:8px; background:#f3f4f6; }
      .meta { color:#666; font-size:13px; }
    </style>
  </head>
  <body>
    <h2>Email Tone Classifier — Demo</h2>
    <p>Type or paste an email and click <b>Classify</b>. The UI uses the saved model + rule overrides.</p>

    <textarea id="emailText">Dear Sir/Madam, I am writing to request an update regarding my application status.</textarea>
    <br>
    <button id="btn" onclick="classify()">Classify</button>

    <div id="out" class="result" style="display:none;">
      <div id="tone" style="font-weight:700"></div>
      <div id="details" class="meta"></div>
    </div>

    <script>
      async function classify(){
        const text = document.getElementById('emailText').value;
        document.getElementById('out').style.display='none';
        document.getElementById('tone').innerText = 'Working...';
        try{
          const res = await fetch('/predict', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({text})
          });
          const js = await res.json();
          document.getElementById('out').style.display='block';
          if(js.error){
            document.getElementById('tone').innerText = 'Error: ' + js.error;
            document.getElementById('details').innerText = '';
            return;
          }
          document.getElementById('tone').innerText = js.tone;
          let meta = 'method: ' + js.method;
          if(js.confidence !== undefined && js.confidence !== null) meta += ' | confidence: ' + (Math.round(js.confidence*100)/100);
          document.getElementById('details').innerText = meta;
        }catch(err){
          document.getElementById('out').style.display='block';
          document.getElementById('tone').innerText = 'Request failed';
          document.getElementById('details').innerText = err.message;
        }
      }
    </script>
  </body>
</html>
"""


# -----------------------------------------------------------------
# README instructions (put this as text or README.md)
# 1) Ensure your virtual env is active
# 2) Install Flask and joblib if not present: pip install flask joblib
# 3) Ensure the model files are at: model/email_tone_model.pkl and model/tfidf_vectorizer.pkl
# 4) Run: python app.py
# 5) Open http://127.0.0.1:5000 in your browser
