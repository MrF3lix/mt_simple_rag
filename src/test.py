import json
import logging
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

from retriever import DenseRetriever, SparseRetriever, OracleRetriever, RandomRetriever, SimilarRetriever, Query, Paragraph
from generator import Generator
from judge import DefaultJudge, LLMJudge

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to the config file", default="config/base.yaml"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    now = datetime.today().strftime('%Y-%m-%d_%H-%M')
    report_path = f"results/{now}"
    Path(report_path).mkdir(parents=True, exist_ok=True)
    logger.debug(f'Started {now}')

    with open(f"{report_path}/config.yaml", 'w') as f:
        OmegaConf.save(cfg, f)

    results = run_test_queries(cfg)

    logger.debug(f'Number of Test Queries:      {len(results)}')
    logger.debug(f'Correct Documents:           {results['correct_document'].sum() / len(results)}')
    logger.debug(f'Correct Paragraph:           {results['correct_paragraph'].sum() / len(results)}')
    logger.debug(f'Correct Answer:              {results['correct_answer'].sum() / len(results)}')

    with open(f'{report_path}/results.json', 'w') as f:
        json.dump(results, f)

def run_test_queries(cfg):
    retriever = load_retriever(cfg)
    generator = Generator(cfg)
    judge = load_judge(cfg)

    results = []
    with open(cfg.documents.target) as f:
        for line in f:
            row = json.loads(line)
            query = Query.model_validate(row, )

            query = retriever.retriev(query)
            query = generator.generate(query)
            query = judge.evaluate(query)

            results.append(query.compute_result())

    return results

def load_judge(cfg):
    if 'judge' in cfg:
        return LLMJudge(cfg)

    return DefaultJudge()

def load_retriever(cfg):
    if cfg.retriever.strategy == 'random':
        return RandomRetriever(cfg)
    elif cfg.retriever.strategy == 'similar':
        return SimilarRetriever(cfg)
    elif cfg.retriever.strategy == 'oracle':
        return OracleRetriever(cfg)
    elif cfg.retriever.strategy == 'sparse':
        return SparseRetriever(cfg)

    return DenseRetriever(cfg)

main()