from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_CACHE = {}

async def load_model(model_name: str):
    if model_name not in MODEL_CACHE:
        try:
            model = AutoModel.from_pretrained(model_name, output_attentions=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            MODEL_CACHE[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "device": device
            }
        except Exception as e:
            raise RuntimeError(f"Model load failed: {str(e)}")
    return MODEL_CACHE[model_name]

@app.websocket("/ws/analyze")
async def websocket_analysis(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            model_name = data.get("model", "bert-base-uncased")
            text = data.get("text", "")
            
            model_data = await load_model(model_name)
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            
            inputs = tokenizer(text, return_tensors="pt").to(model_data["device"])
            outputs = model(**inputs)
            
            # Convert attention to numpy array
            attentions = [attn.cpu().detach().numpy() for attn in outputs.attentions]
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Calculate saliency
            saliency = calculate_saliency(model, inputs)
            
            await websocket.send_json({
                "tokens": tokens,
                "attentions": [a.tolist() for a in attentions],
                "saliency": saliency.tolist()
            })
    except Exception as e:
        print(f"Error: {str(e)}")

def calculate_saliency(model, inputs):
    model.eval()
    embeddings = model.get_input_embeddings()(inputs["input_ids"])
    embeddings.requires_grad = True
    
    outputs = model(inputs_embeds=embeddings)
    loss = outputs.last_hidden_state.mean()
    loss.backward()
    
    saliency = torch.norm(embeddings.grad, dim=2).squeeze(0)
    return saliency.cpu().detach().numpy()

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    UPLOAD_DIR = "backend/models"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    return {"status": "success", "path": file_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
