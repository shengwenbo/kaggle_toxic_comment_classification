
import pandas as pd
from utils.replace import replace as replace_sp
from utils.vocab import *
from utils.tokens import *

if __name__ == "__main__":
    data = pd.read_csv("../data/train_preprocessed.csv")

