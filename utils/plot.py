import matplotlib.pyplot as plt
import numpy as np

def train_valid_loss(train_loss, valid_loss, model_name):
    fig = plt.figure()
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    plt.plot(train_loss,'r',label="train_loss")
    plt.plot(valid_loss,'g',label="valid_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"result/{model_name}.png")

