import joblib
import re

# Load model and vectorizer (names you have in model/)
model = joblib.load("model/email_tone_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# RULES: a list of (pattern, tone) pairs (pattern matched case-insensitive)
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
    (r'\bplease\b', 'polite'),   # careful: 'please' is generic, but helpful
    (r'\bthank you\b', 'appreciative'),
    (r'\bthanks\b', 'appreciative'),
    (r'\bi appreciate\b', 'appreciative'),
    (r'\bwhy\b', 'rude'),        # conservative: presence of 'why' + harsh words => rude
    (r'\bunacceptable\b', 'rude'),
    (r'\bfix it\b', 'rude'),
]

def apply_rules(text):
    t = text.lower()
    for patt, tone in RULES:
        if re.search(patt, t):
            return tone
    return None

def get_model_prediction(text):
    X = vectorizer.transform([text])
    # try to get probability and predicted class
    try:
        probs = model.predict_proba(X)[0]
        classes = model.classes_
        best_idx = probs.argmax()
        return classes[best_idx], probs[best_idx]
    except Exception:
        # fallback if model doesn't support predict_proba
        pred = model.predict(X)[0]
        return pred, None

def main():
    email_text = input("Enter an email text: ").strip()
    if not email_text:
        print("No input provided. Exiting.")
        return

    # 1) quick rule-based check
    rule_tone = apply_rules(email_text)
    if rule_tone:
        print("\nPredicted Tone (rule):", rule_tone)
        return

    # 2) model prediction
    model_pred, confidence = get_model_prediction(email_text)

    # 3) low-confidence guard: if confidence exists and is low, apply small heuristics
    if confidence is not None and confidence < 0.60:
        # low confidence: try some additional checks before trusting model
        # if input contains formal cues, override to formal
        if re.search(r'\bdear\b|\bsir\b|\bmadam\b', email_text.lower()):
            print("\nPredicted Tone (override low-confidence -> formal): formal")
            return
        if re.search(r'\bthank you\b|\bthanks\b|\bi appreciate\b', email_text.lower()):
            print("\nPredicted Tone (override low-confidence -> appreciative): appreciative")
            return
        # otherwise still return the model's prediction but mark confidence
        print(f"\nPredicted Tone (model, low confidence {confidence:.2f}): {model_pred}")
        return

    # 4) confident model result
    print(f"\nPredicted Tone: {model_pred}" + (f"  (confidence {confidence:.2f})" if confidence is not None else ""))

if __name__ == "__main__":
    main()

