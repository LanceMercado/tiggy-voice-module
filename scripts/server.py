
from flask import Flask, request, send_file
from tts.engine import synthesize
import tempfile
import os

app = Flask(__name__)

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return {"error": "Missing 'text'"}, 400
    out_path = tempfile.mktemp(suffix=".wav", dir="output")
    synthesize(text, output_path=out_path)
    return send_file(out_path, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
