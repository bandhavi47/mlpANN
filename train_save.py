import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
import utils.config as config

from utils.model import mlPerceptron
from utils.prepare_data import load_data

logging_str = "[%(asctime)s - %(levelname)s: %(module)s] %(message)s"
logging.basicConfig(filename = os.path.join(config.log_dir,"running_logs.log"),level=logging.INFO, format=logging_str,filemode='a')
os.makedirs(config.log_dir, exist_ok=True)

def trainModel(epochs,filename):
    model_clf = mlPerceptron.model()
    X_valid, X_train, y_valid, y_train, X_test, y_test = load_data()
    VALIDATION = (X_valid, y_valid)
   
    history = model_clf.fit(X_train, y_train, epochs=epochs, validation_data=VALIDATION)

    model_clf.evaluate(X_test, y_test)  
    
    os.makedirs(config.model_dir, exist_ok=True)
    filePath = os.path.join(config.model_dir, filename) 

    model_clf.save(filePath) 

    return history.history

def savePlot(history, filename):
    
    os.makedirs(config.plot_dir, exist_ok=True) 
    plotPath = os.path.join(config.plot_dir, filename)

    pd.DataFrame(history)
    pd.DataFrame(history).plot(figsize=(10,7))
    plt.grid(True)
    plt.savefig(plotPath) 
    
 
if __name__ == "__main__":
    history = trainModel(epochs=config.EPOCHS, filename = config.modelFilename)
    savePlot(history,filename=config.plotFilename)
