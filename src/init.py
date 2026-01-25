import logging
import argparse
from omegaconf import OmegaConf

from knowledge_base import WikiKnowledgeBase, CatechismKnowledgeBase

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

    kb = load_knowledge_base(cfg)
    kb.init_database()
    kb.init_index()

def load_knowledge_base(cfg):
    if 'dataset' in cfg.knowledge_base and cfg.knowledge_base.dataset == 'catechism':
        return CatechismKnowledgeBase(cfg)
    else:
        return WikiKnowledgeBase(cfg)

main()