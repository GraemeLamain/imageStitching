# Faisal Qureshi with help from ChatGPT
# faisal.qureshi@ontariotechu.ca
# Edited and Modified by Graeme Lamain - 100873910

# Start backend
# gunicorn -w 4 -b 127.0.0.1:8000 --timeout 1200 backend:app
# I have just been doing python3 backend.py

from flask import Flask, request, send_file, jsonify, make_response
from flask_cors import CORS
import io
import numpy as np
from PIL import Image

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
    elif "multipart/form-data" in ct:
        print('- multipart/form-data')
        # in case you switch to FormData later
        payload = request.form.to_dict()
        if "image" not in payload and "file" in request.files:
            print('- reading file')
            # file path if you POST a file blob
            f = request.files["file"]
            payload["image"] = "file"
            img = np.array(Image.open(f.stream))
    else:
        print('- json fallback')
        # try JSON as a fallback
        payload = request.get_json(silent=True) or {}

    # Debug: print what we actually got
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
            # 1. Detect & Match features in first two images (SIFT/ORB)
            print(f"Getting key points")
            dts_pts, src_pts, matches = features.get_match_pairs(imgs[0], imgs[1])
            print(f"Found {len(matches)} matches.")

            # 3. Warp and stitch img2
            warped_img = stitching.stitch_img(imgs[0], imgs[1], dts_pts, src_pts)

            # 4. Repeat for all images (not done yet)

            final_result = np.zeros_like(warped_img)

            np.copyto(final_result, warped_img)
            
            # fig = plt.figure()
            # fig.set_size_inches(25,10) 
            # plt.imshow(final_result)
            
            # fig.savefig("stitch_result.png")
            # --- F. Return Result ---
            out = final_result
            
        else:
            out = img
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