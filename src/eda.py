import pandas as pd
import matplotlib.pyplot as plt
import numpy as np








if __name__ == "__main__":
    
    train_df = pd.read_csv('../data/train.zip')
    test_df = pd.read_csv('../data/train.zip')
    print(train_df.info())