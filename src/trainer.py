import argparse
from config import params
from engine import Engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str)
    parser.add_argument('--config', type=str, help='Evaluation config file path')
    args = parser.parse_args()

    engine = Engine(params)
    metrics = engine.train()
    print(metrics)
   

