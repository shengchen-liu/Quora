import os
class DefaultConfigs(object):
    train_data = os.path.abspath("../input/train.csv")  # where is your train data
    test_data =  os.path.abspath("../input/test.csv")  # your test data
    embedding_dir =  os.path.abspath('../input/embeddings')

    logs = "../results/logs/"
    weights = "../results/checkpoints/"
    best_models = "../results/checkpoints/best_models/"
    submit = "../results/submit/"
    model_name = "Baseline"
    lr = 1e-3
    batch_size = 1536
    epochs = 10
    resume = True
    initial_checkpoint = '0'
    gpus = "0"
    fold=5
    model='baseline_pytorch'
    embed_size = 300  # how big is each word vector
    max_features = 120000  # how many unique words to use (i.e num rows in embedding vector)
    maxlen = 72  # max number of words in a question to use
    sample = 0 # for debug
    embed_method = "mean" # concat or mean

config = DefaultConfigs()