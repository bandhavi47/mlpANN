from utils.prepare_data import load_data
from utils.model import mlPerceptron
import numpy as np
import matplotlib.pyplot as plt
import os

def predictValue(X_new,filename):
    model_clf = mlPerceptron.model()
    y_prob = model_clf.predict(X_new)
    y_prob.round(3)
    Y_pred= np.argmax(y_prob, axis=-1)
    plot_dir = "output"
    os.makedirs(plot_dir, exist_ok=True) 
    i=0
    for img_array, pred in zip(X_new, Y_pred):
        plt.imshow(img_array, cmap="binary")
        plt.title(f"predicted: {pred}")
        plt.axis("off")
        plotPath = os.path.join(plot_dir, f"{filename}_{i}")
        plt.savefig(plotPath)
        i+=1
    

if __name__ == "__main__":
    _,_,_,_,X_test,y_test = load_data()
    X_new = X_test[:3]
    predictValue(X_new,"outputValue")