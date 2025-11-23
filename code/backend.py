# Faisal Qureshi with help from ChatGPT
# faisal.qureshi@ontariotechu.ca
# Edited and Modified by Graeme Lamain - 100873910

# Start backend
# gunicorn -w 4 -b 127.0.0.1:8000 --timeout 1200 backend:app


'''
TODO:
- Fix front end to let you input as many images as you want
- Make front end prettier
- Add feature detection and matching to automate the point selection process
- (Optional) Make it so you can detect which image goes where without being given an order


'''

from flask import Flask, request, send_file, jsonify, make_response
from flask_cors import CORS
import io, base64, cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import stitching
import features
import utils

app = Flask(__name__)
CORS(app)

ALLOWED_OPS = {"none", "stitch"}

@app.post("/process")
def process():
    # Robust payload parsing
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

    if "image1" not in payload or "image2" not in payload:
        print('- image not in payload')
        return jsonify(error="missing 'image' field", got=list(payload.keys())), 400

    op = payload.get("op")
    if not op:
        print('- missing op field')
        return jsonify(error="missing 'op' field", got=list(payload.keys())), 400
    op = str(op).lower()
    if op not in ALLOWED_OPS:
        print('- unrecognized op')
        return jsonify(error=f"unknown op '{op}'", allowed=sorted(ALLOWED_OPS)), 400

    # Decode image (unless we already handled file above)
    if payload.get("image1") != "file":
        print('- handling image1')
        try:
            img1 = utils.dataurl_to_numpy(payload["image1"])  # (H,W,C)
        except Exception as e:
            return jsonify(error=f"decode-failed: {e}"), 400
        
    if payload.get("image2") != "file":
        print('- handling image2')
        try:
            img2 = utils.dataurl_to_numpy(payload["image2"])  # (H,W,C)
        except Exception as e:
            return jsonify(error=f"decode-failed: {e}"), 400

    print(f'- img1.shape={img1.shape}')
    print(f'- img2.shape={img2.shape}')

    # ---- simple ops (NumPy) ----
    try:
        # Apply the painting stuff
        if op == "stitch":
            pts_list1 = payload.get("points1","[]")
            pts_list2 = payload.get("points2","[]")

            if len(pts_list1) < 4 or len(pts_list2) < 4:
                return jsonify(error="Need at least 4 points per image"), 400
            
            if len(pts_list1) != len(pts_list2):
                return jsonify(error="Number of points must match"), 400
            
            print(f"Points1: {pts_list1}")
            print(f"Points2: {pts_list2}")

            dst_pts = np.float32([[p['x'], p['y']] for p in pts_list1]) # Reference
            src_pts = np.float32([[p['x'], p['y']] for p in pts_list2]) # To Warp
            
            warped_img = stitching.stitch_img(img1, img2, dst_pts, src_pts)


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
    app.run("127.0.0.1", 8000, debug=True)