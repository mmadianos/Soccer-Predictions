import argparse
import json
from engine import Engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', type=str)
    parser.add_argument('--config', type=str, help='Evaluation config file path')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)

    engine = Engine(config)
    m, mm = engine.train()
    print(m,mm)
   

