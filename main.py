import os
from model.lstm import lstm
from model.attention import attention
import pandas as pd
import utils as my_utils
import argparse
from tensorflow import logging

LABEL_LIST = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]


def preprocess(data_dir="./data"):
    print("begin to preprocess...")
    train_data_path = os.path.join(data_dir, "train.csv")
    new_train_data_path = os.path.join(data_dir, "train_prcssd.csv")
    test_data_path = os.path.join(data_dir, "test.csv")
    new_test_data_path = os.path.join(data_dir, "test_prcssd.csv")
    vocab_path = os.path.join(data_dir, "vocab.txt")
    # 读数据
    logging.info("loading data...")
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    # 预处理
    train_data["tag"] = "train"
    test_data["tag"] = "test"
    data = train_data.append(test_data)
    logging.info("replacing bad words...")
    data["comment_text"] = data.apply(lambda d : my_utils.replace(d["comment_text"]), axis=1)
    logging.info("tokenizing...")
    data["tokens"] = data.apply(lambda d: my_utils.tokenize(d["comment_text"]), axis=1)
    logging.info("making vocabulary...")
    vocab = my_utils.make_vocab(data["tokens"])
    data["tokens"] = data.apply(lambda d: " ".join(d["tokens"]))
    train_data = data[data.tag == "train"]
    test_data = data[data.tag == "test"]
    #保存
    logging.info("saving...")
    train_data.to_csv(new_train_data_path)
    test_data.to_csv(new_test_data_path)
    my_utils.dump_vocab(vocab, vocab_path)
    logging.info("preprocess finished!")

    return train_data, test_data


def load_data(data_dir="./data", if_preprocess=False):
    if if_preprocess:
        preprocess(data_dir)

    logging.info("loading data from %s" % data_dir)
    train_data = pd.read_csv(os.path.join(data_dir, "train_prcssd.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test_prcssd.csv"))

    train_data["tokens"] = train_data.apply(lambda d: d["tokens"].split(" "))
    test_data["tokens"] = test_data.apply(lambda d: d["tokens"].split(" "))

    train_x = train_data["tokens"].values
    train_y = train_data[LABEL_LIST].values
    test_x = test_data["tokens"].values

    return train_x, train_y, test_x, train_data, test_data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode", "-m", type=str)
    arg_parser.add_argument("--preprocess", "-p", type=bool, default=False)
    arg_parser.add_argument("--data", "-d", type=str, default="./data")

    args = arg_parser.parse_args()
    mode = args.mode
    if_preprocess = args.preprocess
    data_dir = args.data

    logging.set_verbosity(logging.INFO)

    train_x, train_y, test_x, train_data, test_data = load_data(data_dir=data_dir, if_preprocess=if_preprocess)

    logging.info("building model...")
    model = attention()
    restored = model.restore()

    if mode == "train":
        logging.info("training...")
        model.train(train_x, train_y,epochs=100,batch_size=150)
    elif mode == "evaluate":
        logging.info("evaluating...")
        if restored:
            for name, value in model.evaluate(train_x,train_y,batch_size=150):
                print("name: %s, value: %f" % (name, value))
        else:
            logging.error("error: model weights not exist!")
    elif mode == "submit":
        logging.info("predicting final result...")
        test_data[LABEL_LIST] = model.predict(test_x, batch_size=150)
        test_data = test_data[["id"]+LABEL_LIST]
        test_data.to_csv("submission.csv", index=False)