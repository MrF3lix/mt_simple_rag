from fastapi import FastAPI
from omegaconf import OmegaConf

from retriever import Query, DenseRetriever
from generator import Generator

cfg = OmegaConf.load("config/base.yaml")

app = FastAPI()
retriever = DenseRetriever(cfg)
generator = Generator(cfg)

@app.get("/")
def status():
    return "running"

@app.post("/query")
def query(query: Query) -> Query:
    query = retriever.retriev(query)
    query = generator.generate(query)

    return query
