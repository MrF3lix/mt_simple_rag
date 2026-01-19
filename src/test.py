import logging
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from datetime import datetime

from retriever import DenseRetriever, Query, Paragraph

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

    retriever = DenseRetriever(cfg)

    df_q = pd.read_json(cfg.documents.target, lines=True)
    results = []
    for _, row in tqdm(df_q.iterrows(), total=df_q.shape[0]):
        query = Query(
            id = row['id'],
            input=row['input'],
            references=extract_wikipedia_link(row),
            retrieved=retriever.retriev(row['input'])
        )

        results.append(query.compute_result())

    results = pd.DataFrame(results)

    logger.debug(f'Number of Test Queries:      {len(results)}')
    logger.debug(f'Correct Documents:           {results['correct_document'].sum() / len(results)}')
    logger.debug(f'Correct Paragraph:           {results['correct_paragraph'].sum() / len(results)}')

    results.to_json(f'{report_path}/results.json', orient='records')

def extract_wikipedia_link(row) -> list[Paragraph]:
    paragraphs = []
    for item in row['output'][0]['provenance']:
        p = Paragraph(
            document_id=item['wikipedia_id'],
            index=item['start_paragraph_id'] + 1
        )
        paragraphs.append(p)

    return paragraphs


main()