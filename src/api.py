import argparse
from fastapi import FastAPI
from omegaconf import OmegaConf

from retriever import Query, DenseRetriever
from generator import Generator

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--config", type=str, help="Path to the config file", default="config/base.yaml"
# )
# args = parser.parse_args()

cfg = OmegaConf.load("config/base.yaml")

app = FastAPI()
retriever = DenseRetriever(cfg)
generator = Generator(cfg)

@app.get("/")
def status():
    return "running"

@app.post("/query")
async def query(query: Query) -> Query:
    query = retriever.retriev(query)
    query = generator.generate(query)

    return query
