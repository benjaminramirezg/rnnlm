cd ../src/main/python
python predict.py --data=../../../data/corpus/corpus.txt\
                  --config=../../../config/config.json\
                  --model=../../../models/model.torch\
                  --vocabulary=../../../models/vocabulary.pkl\
                  --output=../../../output/corpus-probabilities.tsv