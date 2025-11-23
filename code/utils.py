import numpy as np
import base64, io
from PIL import Image

def dataurl_to_numpy(data_url: str) -> np.ndarray:
    print('- dataurl_to_numpy')
    if data_url.startswith("data:"):
        print('- startswith data')
        _, b64 = data_url.split(",", 1)
    else:
        print('- do not startwith data')
        b64 = data_url
    # normalize + pad
    print('- decoding b64')
    b64 = b64.strip().replace("\n","").replace("\r","").replace(" ","+")
    pad = len(b64) % 4
    if pad: b64 += "=" * (4 - pad)
    raw = base64.b64decode(b64)
    
    # Force conversion to RGB to drop the Alpha channel (RGBA -> RGB)
    pil_image = Image.open(io.BytesIO(raw)).convert('RGB') 
    arr = np.array(pil_image)

    print(f'- arr.ndim = {arr.ndim}')
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr  # uint8 (H,W,C)


def numpy_to_png(arr: np.ndarray) -> bytes:
    print('- numpy_to_png')
    print(f'- arr.ndim = {arr.ndim}')
    if arr.ndim != 2:
        print(f'- arr.shape[2]={arr.shape[2]}')
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    elif arr.shape[2] == 1:
        img = Image.fromarray(arr[...,0], mode="L")
    elif arr.shape[2] == 3:
        img = Image.fromarray(arr, mode="RGB")
    elif arr.shape[2] == 4:
        img = Image.fromarray(arr, mode="RGBA")
    else:
        raise ValueError("Unsupported channel count")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()