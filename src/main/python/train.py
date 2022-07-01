import json
import pickle
import argparse
from rnnlm import LanguageModelTrainer, Vocabulary, LanguageModelDataset

parser = argparse.ArgumentParser()  
parser.add_argument("-t", "--traindata", help="Path to folder with data to train the language model", type=str, required=True)
parser.add_argument("-e", "--evaldata", help="Path to folder with data to eval the language model", type=str, required=True)
parser.add_argument("-c", "--config", help="Path to JSON file with hyperparameters to train the language model", type=str, required=True)
parser.add_argument("-m", "--model", help="Path to output file where language model will be saved", type=str, required=True)
parser.add_argument("-p", "--vocabulary", help="Path to output pickle file where vocabulary will be saved", type=str, required=True)
args = parser.parse_args()

_DEFAULT_CONFIG = {
    "BATCH_SIZE": 16,
    "NUM_EPOCHS": 2,
    "EMBEDDING_DIM": 10,
    "DROPOUT": 0.2,
    "RNN_DIM": 50,
    "RNN_LAYERS": 2,
    "MAX_LEN": 50
    }

print("Reading config from {}...".format(args.config))
config = None
with open(args.config, "r") as fh:
    config = json.loads(fh.read())
    for default_key, default_value in _DEFAULT_CONFIG.items():
        if default_key not in config:
            config[default_key] = default_value

print("Loading dataset from {}...".format(args.traindata))
vocabulary = Vocabulary(
    path=args.traindata
    )

train_dataset = LanguageModelDataset(
    args.traindata, vocabulary=vocabulary, max_len=config["MAX_LEN"]
    )

eval_dataset = LanguageModelDataset(
    args.evaldata, vocabulary=vocabulary, max_len=config["MAX_LEN"]
    )

print("Creating model with config: {}...".format(config))
lm_trainer = LanguageModelTrainer(
    config=config,
    vocabulary=vocabulary
)

print("Training...")
lm_trainer.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
    )

print("Saving model in {}...".format(args.model))
lm_trainer.save(
    args.model
    )

print("Saving vocabulary in {}...".format(args.vocabulary))
with open(args.vocabulary, 'wb') as handle:
    pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
