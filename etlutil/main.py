from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os

app = FastAPI()

FILE_MAPPING = {
    "barito_router_readme": "datasets/documents/datasets/BaritoLog/barito-router/README.md",
    "wiki_readme": "datasets/documents/datasets/BaritoLog/wiki/readme.md", 
    "gopay_article": "datasets/documents/datasets/GoPay.sh_ A glimpse into Indonesia\u2019s leading e-wallet GoPay\u2019s Developer Experience - The Jakarta Post.pdf",
    "barito_logging": "datasets/documents/datasets/How we built \u2018BARITO\u2019 to enhance logging - 5 min read.pdf"
}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/feishu/drive/files/{file_token}/download")
async def download_file(file_token: str):
    if file_token not in FILE_MAPPING:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = FILE_MAPPING[file_token]
    print(f"Looking for file: {file_path}")
    print(f"Exists: {os.path.exists(file_path)}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type='application/octet-stream')


@app.get("/feishu/drive/list-files")
async def list_files():
    return {"files": list(FILE_MAPPING.keys())}
