"""
FastAPI server for SAE feature visualization.

Usage:
    pip install fastapi uvicorn
    uvicorn interpretability.app.main:app --reload
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import TextIteratorStreamer

from interpretability.inference import FeatureSteerer
from interpretability.models.olmo2_1b import load_model
from interpretability.nn.activations import ActivationCollector
from interpretability.nn.sae import SAEConfig, SparseAutoencoder

CHECKPOINT = "olmo2_1b_sae_layer8.pt"
ANALYSIS_FILE = "feature_analysis.json"
STATIC_DIR = Path(__file__).parent / "static"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, tokenizer = load_model()
    model.eval()

    sae = SparseAutoencoder(SAEConfig()).to(model.device)
    sae.load_state_dict(torch.load(CHECKPOINT, map_location=model.device, weights_only=True))
    sae.eval()

    analysis_path = Path(ANALYSIS_FILE)
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}

    _state.update(model=model, tokenizer=tokenizer, sae=sae, analysis=analysis)
    print(f"Loaded SAE · {len(analysis):,} features with pre-computed examples")
    yield
    _state.clear()


app = FastAPI(title="SAE Feature Explorer", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str


class SteerFeature(BaseModel):
    idx: int
    scale: float


class SteerRequest(BaseModel):
    prompt: str
    features: list[SteerFeature]
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    layer_idx: int = 8


@app.get("/api/features/{feature_idx}")
def get_feature(feature_idx: int):
    if not 0 <= feature_idx < SAEConfig().dict_size:
        raise HTTPException(status_code=400, detail=f"Feature index must be 0–{SAEConfig().dict_size - 1}")
    examples = _state["analysis"].get(str(feature_idx), [])
    return {"idx": feature_idx, "examples": examples}


@app.post("/api/analyze")
def analyze_text(req: AnalyzeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    model = _state["model"]
    tokenizer = _state["tokenizer"]
    sae = _state["sae"]

    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    token_ids = inputs["input_ids"][0].tolist()

    with ActivationCollector(model.model.layers[8]) as collector:
        with torch.no_grad():
            model(**inputs, output_hidden_states=False)
        hidden = collector.pop()[0].squeeze(0).float().to(model.device)

    features = sae(hidden).features

    tokens = []
    for tok_id, feat_row in zip(token_ids, features):
        active = feat_row.nonzero(as_tuple=False).squeeze(-1).tolist()
        top_features = sorted(
            [{"idx": int(i), "activation": round(float(feat_row[i]), 4)} for i in active],
            key=lambda x: x["activation"],
            reverse=True,
        )[:20]
        tokens.append(
            {
                "token": tokenizer.decode([tok_id], skip_special_tokens=False),
                "token_id": tok_id,
                "max_activation": round(float(feat_row.max()), 4),
                "features": top_features,
            }
        )

    return {"tokens": tokens}


@app.post("/api/steer")
def steer_generate(req: SteerRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if not req.features:
        raise HTTPException(status_code=400, detail="At least one feature required")
    dict_size = SAEConfig().dict_size
    for f in req.features:
        if not 0 <= f.idx < dict_size:
            raise HTTPException(status_code=400, detail=f"Feature index {f.idx} out of range 0–{dict_size - 1}")

    def _stream():
        model = _state["model"]
        tokenizer = _state["tokenizer"]
        sae = _state["sae"]

        messages = [{"role": "user", "content": req.prompt}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": req.max_new_tokens,
            "do_sample": True,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "streamer": streamer,
        }

        steerer = FeatureSteerer(model, sae, req.layer_idx)
        for f in req.features:
            steerer.set_feature(f.idx, f.scale)

        with steerer:
            thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
            thread.start()
            for token in streamer:
                yield f"data: {json.dumps({'token': token})}\n\n"
            thread.join()

        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
