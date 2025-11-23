(() => {
  const upload1   = document.getElementById('upload1');
  const loadBtn1  = document.getElementById('loadBtn1');
  const canvas1   = document.getElementById('canvas1');
  const ctx1      = canvas1.getContext('2d');
  const wrapper1  = canvas1.parentElement;

  const upload2   = document.getElementById('upload2');
  const loadBtn2  = document.getElementById('loadBtn2');
  const canvas2   = document.getElementById('canvas2');
  const ctx2      = canvas2.getContext('2d');
  const wrapper2  = canvas2.parentElement;

  const canvasStitch  = document.getElementById('canvasStitch');
  const ctxStitch     = canvasStitch.getContext('2d');
  const wrapperStitch = canvasStitch.parentElement;

  const undoBtn     = document.getElementById('undoPoint');
  const resetBtn    = document.getElementById('reset');
  const savePng     = document.getElementById('savePng');
  const sendToPyBtn = document.getElementById('sendToPy');

  let img1 = null;
  let img2 = null;
  let imgStitched = null;

  // Coordinate arrays (img coordinates, not screen coordinates)
  let points1 = [];
  let points2 = [];
  let lastPointOnImg1 = true;   // just so i know for undoing

  // let img = null;
  // let imgAfter = null;
  let scale = 1;
  let offset = { x: 0, y: 0 };

  let isDown = false;
  let start = { x: 0, y: 0 };
  let startOffset = { x: 0, y: 0 };

  // Specify the canvas and wrapper you are working on then pass the draw function you want to run
  function resizeCanvas(canvas, wrapper, drawFn) {
    const r = wrapper.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = r.width * dpr;
    canvas.height = r.height * dpr;

    canvas.style.width = r.width + 'px';
    canvas.style.height = r.height + 'px';
    if (drawFn) drawFn();
  }
  new ResizeObserver(() => resizeCanvas(canvas1, wrapper1, drawImage(ctx1, canvas1, img1, points1))).observe(wrapper1);
  new ResizeObserver(() => resizeCanvas(canvas2, wrapper2, drawImage(ctx2, canvas2, img2, points2))).observe(wrapper2);
  new ResizeObserver(() => resizeCanvas(canvasStitch, wrapperStitch, drawStitched)).observe(wrapperStitch);


  function fitToView(canvas, image) {
    if(!image) return { s: 1, tx: 0, ty: 0 };
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.width / dpr;
    const ch = canvas.height / dpr;


    const s = Math.min(cw / image.naturalWidth, ch / image.naturalHeight) * 0.9;
    const tx = (cw - image.naturalWidth * s) / 2;
    const ty = (ch - image.naturalHeight * s) / 2;
    return { s, tx, ty };
  }

  function drawImage(ctx, canvas, img, points) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#000'; 
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.scale(dpr, dpr);  // Scale context to work in logical pixels

    if (!img) return;
    const { s, tx, ty } = fitToView(canvas, img);

    // ctx.save();
    ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight, tx, ty, img.naturalWidth*s, img.naturalHeight*s);
    points.forEach((p, i) => {
          const px = tx + p.x * s;
          const py = ty + p.y * s;
          
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, Math.PI*2);
          ctx.fillStyle = 'red';
          ctx.fill();
          ctx.lineWidth = 2;
          ctx.strokeStyle = 'white';
          ctx.stroke();
          
          ctx.fillStyle = 'white';
          ctx.font = '14px Arial';
          ctx.fillText(i+1, px + 8, py - 8);
      });
    ctx.restore();
  }

  function drawStitched() {
    ctxStitch.clearRect(0, 0, canvasStitch.width, canvasStitch.height);
    ctxStitch.fillStyle = '#000';
    ctxStitch.fillRect(0, 0, canvasStitch.width, canvasStitch.height);

    if (!imgStitched) return;

    const dpr = window.devicePixelRatio || 1;
    ctxStitch.save();
    ctxStitch.scale(dpr, dpr);
    // const cw = canvasStitch.width / dpr;
    // const ch = canvasStitch.height / dpr;
    const {s, tx, ty} = fitToView(canvasStitch, imgStitched);

    // ctxStitch.translate(cw/2 + offset.x, ch/2 + offset.y);
    // ctxStitch.scale(scale, scale);
    // ctxStitch.imageSmoothingEnabled = true;  
    // ctxStitch.imageSmoothingQuality = 'high';
    // ctxStitch.drawImage(imgStitched, -imgStitched.naturalWidth/2, -imgStitched.naturalHeight/2);
    ctxStitch.drawImage(imgStitched, 0, 0, imgStitched.naturalWidth, imgStitched.naturalHeight, tx, ty, imgStitched.naturalWidth*s, imgStitched.naturalHeight*s);
    ctxStitch.restore();
  }

  function handleClick(e, canvas, img, points, ctx) {
    if (!img) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const { s, tx, ty } = fitToView(canvas, img); 

    const imgX = (clickX - tx) / s;
    const imgY = (clickY - ty) / s;

    // Ensure click is within image bounds
    if(imgX > 0 && imgX <= img.naturalWidth && imgY > 0 && imgY <= img.naturalHeight) {
      points.push({x: imgX, y: imgY});
      if (img === img1) lastPointOnImg1 = true;
      else lastPointOnImg1 = false;
      drawImage(ctx, canvas, img, points);
    }
  }

  canvas1.addEventListener('pointerup', e => handleClick(e, canvas1, img1, points1, ctx1));
  canvas2.addEventListener('pointerup', e => handleClick(e, canvas2, img2, points2, ctx2));


  // ===== Interactions
  loadBtn1.onclick = () => upload1.click();
  upload1.onchange = e => { 
    const f = e.target.files?.[0]; 
    if (f) {
      const reader = new FileReader();
      reader.onload = e => {
        const im = new Image();
        im.onload = () => {
          img1 = im;
          drawImage(ctx1, canvas1, img1, points1);
        }
        im.src = e.target.result;
      }
      reader.readAsDataURL(f);
    }
  };

  loadBtn2.onclick = () => upload2.click();
  upload2.onchange = e => { 
    const f = e.target.files?.[0]; 
    if (f) {
      const reader = new FileReader();
      reader.onload = e => {
        const im = new Image();
        im.onload = () => {
          img2 = im;
          drawImage(ctx2, canvas2, img2, points2);
        }
        im.src = e.target.result;
      }
      reader.readAsDataURL(f);
    }
  };

  // Do this only on the after since the before hasnt changed
  resetBtn.onclick = () => {
    points1 = [];
    points2 = [];
    drawImage(ctx1, canvas1, img1, points1);
    drawImage(ctx2, canvas2, img2, points2);
    imgStitched = null;
    scale = 1;
    offset = { x: 0, y: 0};
    drawStitched();
  };

  savePng.onclick = () => {
    if (!imgStitched) return;
    const a = document.createElement('a');
    a.download = 'stitched.png';
    a.href = canvasStitch.toDataURL('image/png');
    a.click();
  };

  // Changed all the event listeners to only work for the after
  // I could do this for canvas1 as well but it would just be copying code
  // canvasStitch.addEventListener('wheel', (e) => {
  //   if(!imgStitched) return;
  //   e.preventDefault();
  //   const delta = -Math.sign(e.deltaY) * 0.08;
  //   const newScale = Math.max(0.05, Math.min(10, scale * (1 + delta)));

  //   const rect = canvasStitch.getBoundingClientRect();
  //   const dpr = window.devicePixelRatio || 1;
  //   const mx = (e.clientX - rect.left) * dpr;
  //   const my = (e.clientY - rect.top) * dpr;
  //   const cx = canvasStitch.width/2 + offset.x;
  //   const cy = canvasStitch.height/2 + offset.y;

  //   const dx = mx -cx, dy = my - cy;
  //   const ratio = newScale / scale;
  //   offset.x -= dx * (ratio - 1);
  //   offset.y -= dy * (ratio -1);
  //   scale = newScale;
  //   drawStitched();
  // });

  // canvasStitch.addEventListener('pointerdown', (e) => {
  //   if (!imgStitched) return;
  //   isDown = true;
  //   const r = canvasStitch.getBoundingClientRect(); 
  //   const dpr = window.devicePixelRatio || 1;
  //   start.x = (e.clientX - r.left) * dpr;
  //   start.y = (e.clientY - r.top) * dpr;
  //   startOffset = { ...offset };
  // });

  // canvasStitch.addEventListener('pointermove', (e) => {
  //   if (!isDown || !imgStitched) return;
  //   const r = canvasStitch.getBoundingClientRect(); 
  //   const dpr = window.devicePixelRatio || 1;
  //   const x = (e.clientX - r.left) * dpr;
  //   const y = (e.clientY - r.top) * dpr;

  //   offset.x = startOffset.x + (x - start.x);
  //   offset.y = startOffset.y + (y - start.y);
  //   drawStitched();
  // });
  window.addEventListener('pointerup', () => { isDown = false; });

  undoBtn.onclick = () => {
    if (lastPointOnImg1) {
      points1.pop();
      drawImage(ctx1, canvas1, img1, points1);
    } else {
      points2.pop();
      drawImage(ctx2, canvas2, img2, points2);
    }
  };


  // ===== ADDED: Send canvas → Python → get processed PNG → draw back
  async function sendToPython() {
    if (!img1 || !img2) {
      alert("Load both images to stitch.");
      return;
    }

    if(points1.length < 4 || points2.length < 4) {
      alert("Select at least 4 points on each image.");
      return;   
    }

    if(points1.length !== points2.length) {
      alert("Number of points on both images must be the same.");
      return; 
    }

    sendToPyBtn.textContent = "Stitching...";
    sendToPyBtn.disabled = true;

    // Get a PNG data URL of what you currently see on the canvas
    // const dataURL1 = canvas1.toDataURL("image/png");
    // const dataURL2 = canvas2.toDataURL("image/png");

    const res = await fetch("http://127.0.0.1:8000/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        image1: img1.src, 
        image2: img2.src,
        points1: points1,
        points2: points2,
        op: "stitch"
      })
    });

    if (!res.ok) {
      const msg = await res.text();
      alert("Server error: " + msg);
      sendToPyBtn.textContent = "Stitch";
      sendToPyBtn.disabled = false;
      return;
    }

    // Response is an image/png; turn it into an <img> and adopt it as the new base image
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const processed = new Image();
    processed.onload = () => {
      imgStitched = processed;
      // Refit the after image and run the draw function
      scale = Math.min(canvasStitch.width / processed.width, canvasStitch.height / processed.height) * 0.9;
      offset = {x:0, y:0};
      sendToPyBtn.textContent = "Stitch";
      sendToPyBtn.disabled = false;
      resizeCanvas(canvasStitch, wrapperStitch, drawStitched);

      URL.revokeObjectURL(url);
    };
    processed.src = url;
  }
  sendToPyBtn.onclick = () => { sendToPython().catch(err => alert(err)); };
  // ===== /ADDED

  resizeCanvas(canvas1, wrapper1, drawImage(ctx1, canvas1, img1, points1));
  resizeCanvas(canvas2, wrapper2, drawImage(ctx2, canvas2, img2, points2));
  resizeCanvas(canvasStitch, wrapperStitch, drawStitched);
})();