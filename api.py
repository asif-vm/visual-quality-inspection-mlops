from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from src.model import DefectCNN


ARTIFACT = Path(__file__).parent / "artifacts" / "model.pt"
app = FastAPI(title="Visual Quality Inspection API", version="1.0.0")
model = DefectCNN()
ready = False
if ARTIFACT.exists():
    bundle = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    ready = True


@app.get("/", response_class=HTMLResponse)
def inspection_page() -> str:
    return PAGE


@app.get("/health")
def health():
    return {"model_ready": ready}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    if not ready:
        raise HTTPException(503, "Run train.py first")
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(415, "Upload a PNG or JPEG image")
    try:
        image = Image.open(BytesIO(await file.read())).convert("L").resize((32, 32))
    except Exception as exc:
        raise HTTPException(400, "Invalid image") from exc
    values = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.tensor(values).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    defect_probability = float(probabilities[1])
    return {
        "classification": "defect" if defect_probability >= 0.5 else "normal",
        "defect_probability": round(defect_probability, 4),
        "review_required": 0.4 <= defect_probability <= 0.6,
    }


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Visual Quality Inspector</title><style>
*{box-sizing:border-box}body{margin:0;background:#09111f;color:#eef4ff;font:15px system-ui}header{text-align:center;padding:42px 20px}header h1{font-size:34px;margin:0 0 9px}.muted{color:#9badc5}.shell{max-width:920px;margin:auto;padding:0 24px 40px}.panel{background:#121e30;border:1px solid #243754;border-radius:18px;padding:28px;box-shadow:0 16px 40px #0004}.drop{display:block;border:2px dashed #48647f;border-radius:14px;text-align:center;padding:38px 20px;cursor:pointer;transition:.2s}.drop:hover,.drop.over{border-color:#57d3bd;background:#122a33}.drop input{display:none}button{border:0;border-radius:9px;padding:12px 20px;background:#57d3bd;color:#07151b;font-weight:800;cursor:pointer;margin-top:18px;width:100%}button:disabled{opacity:.4}.preview{display:none;grid-template-columns:220px 1fr;gap:26px;align-items:center;margin-top:25px}.preview img{width:220px;height:220px;object-fit:cover;border-radius:12px;background:#080d15}.result{padding:20px;border-radius:12px;background:#0b1524}.badge{display:inline-block;padding:6px 12px;border-radius:20px;font-weight:800}.normal{background:#123d35;color:#73efce}.defect{background:#4b2025;color:#ff9ea6}.review{background:#453b17;color:#ffd96c}.meter{height:12px;background:#26344a;border-radius:9px;overflow:hidden;margin:13px 0}.meter div{height:100%;background:linear-gradient(90deg,#57d3bd,#ff6b78)}.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px}.step{background:#101a2a;border-radius:12px;padding:16px}.step b{color:#57d3bd}@media(max-width:650px){.preview,.steps{grid-template-columns:1fr}.preview img{width:100%;height:260px}}</style></head><body><header><h1>Visual Quality Inspector</h1><div class="muted">Upload a surface image to screen for a scratch or dark spot.</div></header><main class="shell"><section class="panel"><label class="drop" id="drop"><input type="file" id="file" accept="image/png,image/jpeg"><strong>Drop a PNG or JPEG here</strong><p class="muted">or click to choose a close-up surface image</p></label><button id="inspect" disabled>Inspect image</button><div class="preview" id="preview"><img id="image"><div class="result" id="result"><span class="muted">Ready to inspect</span></div></div></section><section class="steps"><div class="step"><b>1. Prepare</b><p>Use a clear, close-up photo of a mostly plain surface.</p></div><div class="step"><b>2. Inspect</b><p>The model converts it to grayscale and checks its visual pattern.</p></div><div class="step"><b>3. Decide</b><p>Normal items pass; defects or uncertain results receive review.</p></div></section></main><script>
let input=document.querySelector('#file'),drop=document.querySelector('#drop'),button=document.querySelector('#inspect'),preview=document.querySelector('#preview'),image=document.querySelector('#image'),result=document.querySelector('#result'),selected;function choose(f){if(!f)return;selected=f;image.src=URL.createObjectURL(f);preview.style.display='grid';button.disabled=false;result.innerHTML='<span class="muted">Ready to inspect</span>'}input.onchange=e=>choose(e.target.files[0]);['dragenter','dragover'].forEach(x=>drop.addEventListener(x,e=>{e.preventDefault();drop.classList.add('over')}));['dragleave','drop'].forEach(x=>drop.addEventListener(x,e=>{e.preventDefault();drop.classList.remove('over')}));drop.ondrop=e=>choose(e.dataTransfer.files[0]);button.onclick=async()=>{button.disabled=true;button.textContent='Inspecting…';let form=new FormData();form.append('file',selected);let r=await fetch('/inspect',{method:'POST',body:form}),d=await r.json();if(r.ok){let pct=Math.round(d.defect_probability*100),label=d.review_required?'Needs human review':d.classification==='defect'?'Possible defect':'Looks normal',klass=d.review_required?'review':d.classification;result.innerHTML=`<span class="badge ${klass}">${label}</span><h2>${pct}% defect probability</h2><div class="meter"><div style="width:${pct}%"></div></div><p class="muted">${d.review_required?'The model is uncertain, so a person should inspect this item.':d.classification==='defect'?'The item should be separated for a quality check.':'The image did not show the defect patterns learned by this demo model.'}</p>`}else result.textContent=d.detail||'Inspection failed';button.disabled=false;button.textContent='Inspect image'};</script></body></html>'''

