# Main Backend Endpoint
# Graeme Lamain - 100873910

# Start backend  - gunicorn -w 4 -b 127.0.0.1:8000 --timeout 1200 backend:app
# Start frontend - python3 -m http.server 8001

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import io
import numpy as np
from matplotlib import pyplot as plt

import stitching
import features
import utils

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_OPS = {"none", "stitch"}

@app.post("/process")
def process():
    # Robust payload parsing
    print("Request from frontend received.")
    ct = request.headers.get("Content-Type", "")
    payload = None
    if "application/json" in ct:
        print('- application/json')
        payload = request.get_json(silent=True) or {}
    else:
        print('- json fallback')
        # try JSON as a fallback
        payload = request.get_json(silent=True) or {}

    print("CT:", ct)
    print("Keys:", list(payload.keys()))
    print("op:", payload.get("op"))

    op = payload.get("op")
    if not op:
        print('- missing op field')
        return jsonify(error="missing 'op' field", got=list(payload.keys())), 400
    op = str(op).lower()
    if op not in ALLOWED_OPS:
        print('- unrecognized op')
        return jsonify(error=f"unknown op '{op}'", allowed=sorted(ALLOWED_OPS)), 400
    
    if "images" not in payload:
        print('- images not in payload')
        return jsonify(error="missing 'images' field", got=list(payload.keys())), 400
    
    image_data_list = payload.get("images", [])

    if not image_data_list or not isinstance(image_data_list, list):
        return jsonify(error="missing or invalid 'images' list"), 400
    
    print(f"- received {len(image_data_list)} images for stitching")

    imgs = []

    try:
        for idx, img_str in enumerate(image_data_list):
            img_np = utils.dataurl_to_numpy(img_str)
            imgs.append(img_np)
    except Exception as e:
        return jsonify(error=f"decode-failed at index {len(imgs)}: {e}"), 400
    
    if len(imgs) < 2:
        return jsonify(error="Need at least 2 images to stitch"),  400

    # ---- simple ops (NumPy) ----
    try:
        out = None
        if op == "stitch":
            print(f"========== Start ==========")

            # Get Key points for all N images
            # Uses SIFT to detect points and get their descriptors
            print(f"========== Key Points ==========")
            key_points = features.get_key_points(imgs)

            # Match features on each image to others
            # Uses KNN matching to find the 'k' best matches for every feature in one image, in the other
            # This will then check if the feature is unique (distance between best and second best is very high)
            # Keeps the unique ones
            print(f"========== Matching Points ==========")
            valid_matches = features.get_matches(imgs, key_points)

            # Compute all the homographies needed
            print(f"========== Chain Homography ==========")
            H_matrices = stitching.chain_homographies(valid_matches, key_points)

            # Get Global Homographies: Chain all matricies to a single anchor image
            print(f"========== Global Homography ==========")
            global_H = stitching.global_homography(H_matrices, len(imgs))
            
            # Warp and paste
            print(f"========== Warping ==========")
            panorama = stitching.warp_and_stitch(global_H, key_points)

            final_result = np.zeros_like(panorama)
            np.copyto(final_result, panorama)
            out = final_result

            plt.imsave("demo.jpg", out)
            print(f"========== End ==========")
        else:
            out = None
    except Exception as e:
        return jsonify(error=f"op-failed: {e}"), 500

    png = utils.numpy_to_png(out)
    # Had to change this part so that we can communicate the histogram back and forth
    buf = io.BytesIO(png)
    # Make the response the image (buf value)
    response = make_response(buf.getvalue())
    response.mimetype = "image/png"
    print(f"Response: {response.headers.getlist}")
    return response
    
# Optional echo endpoint to debug client payloads quickly
@app.post("/echo")
def echo():
    return {"content_type": request.headers.get("Content-Type",""),
            "json": request.get_json(silent=True),
            "form_keys": list(request.form.keys()),
            "files": list(request.files.keys())}

if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)