from fxtrade.optimize.trainer import Trainer
import json

if __name__ == "__main__":
    config = json.load(open("config.json", "r"))
    trainer = Trainer(config)
    trainer.run()