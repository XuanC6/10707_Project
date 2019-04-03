# -*- coding: utf-8 -*-
import time
import numpy as np
import sys, os, glob
import tensorflow as tf

# from trainer import Trainer
from trainer_3 import Trainer
from config import Configuration

tf.enable_eager_execution()

'''
Execute training
'''

if __name__ == "__main__":
    Config = Configuration()
    MyTrainer = Trainer(Config)
    print("Building Model Succeed")
    # print(len(MyTrainer.agent.variables))
    MyTrainer.train()
    MyTrainer.plot_test_result()
    # print(MyTrainer.agent.weights)
    
