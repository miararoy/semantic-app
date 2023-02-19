from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
import uvicorn 
import joblib
import openai
import pandas as pd

app = FastAPI()
pca = joblib.load('artifacts/pca.joblib')

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def embed_and_pca(text, pca):
    embedding = pd.DataFrame(data=[get_embedding(text)], columns=[f"dim_{i}" for i in range(1536)])
    pca_data = pca.transform(embedding)[0]
    data = {
        "x": (pca_data[0] + 0.2502908642012125)   / (0.3087829964094858 + 0.2502908642012125),
        "y": (pca_data[1] + 0.1771318544789371)  / (0.37430858897019964 + 0.1771318544789371),
        "z": (pca_data[2] + 0.22683464398253872) / (0.23989997555815173 + 0.22683464398253872)
    }
    return data

@app.get("/")
async def root():
    return {"message": "hello world"}

@app.get("/get_coordinates")
async def get_coordinates(text: str):
    coor = embed_and_pca(text, pca=pca)
    retults = {
        "coordinates": coor,
        "text": text,
    }
    return retults

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)