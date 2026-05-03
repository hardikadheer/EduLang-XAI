"""
Flask API server that wraps the EduLang seq2seq translation model.
Run this alongside the Express app to enable ML-powered translations.

Usage: python ml/model/api.py
Runs on: http://localhost:5000
"""

from flask import Flask, request, jsonify
from infer import translate

app = Flask(__name__)


@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    from_lang = data.get("from", "english").lower()
    to_lang = data.get("to", "french").lower()

    # Currently only English -> French is supported
    if from_lang == "english" and to_lang == "french":
        result = translate(text)
    elif from_lang == "french" and to_lang == "english":
        result = f"[French → English not yet supported] {text}"
    else:
        result = f"[{from_lang} → {to_lang} not yet supported] {text}"

    return jsonify({
        "original": text,
        "translated": result,
        "from": from_lang,
        "to": to_lang
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "edulang_seq2seq"})


if __name__ == "__main__":
    print("EduLang ML API starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
