import json
import random

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/process", methods=["POST"])
def process_frame():
    data = request.get_json()
    with open("payload.json", "w") as f:
        f.write(json.dumps(data))

    return (
        jsonify(
            {
                "x": random.randint(100, 500),
                "y": random.randint(100, 500),
                "label": "Random Point",
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
