import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from utils.model import mlPerceptron
from utils.prepare_data import load_data

def trainModel(epochs,filename):
    model_clf = mlPerceptron.model()
    X_valid, X_train, y_valid, y_train, X_test, y_test = load_data()
    VALIDATION = (X_valid, y_valid)
   
    history = model_clf.fit(X_train, y_train, epochs=epochs, validation_data=VALIDATION)

    model_clf.evaluate(X_test, y_test)  

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    filePath = os.path.join(model_dir, filename) 

    model_clf.save(filePath) 

    return history.history

def savePlot(history, filename):
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True) 
    plotPath = os.path.join(plot_dir, filename)

    pd.DataFrame(history)
    pd.DataFrame(history).plot(figsize=(10,7))
    plt.grid(True)
    plt.savefig(plotPath) 
    
 
if __name__ == "__main__":
    EPOCHS = 30
    history = trainModel(epochs=EPOCHS, filename="model.h5")
    savePlot(history,filename="plot.png")
