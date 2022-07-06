from scipy.io import arff
import pandas as pd
from GAN2 import GAN
from Dataset import Dataset


data = Dataset('Assignment 4 files/german_credit.arff')
data.norm()
gan = GAN(32, 0.0002, 100, data.norm_df.shape[1])
gan.train(data.norm_df, 'german_credit', 200+1, 10)
