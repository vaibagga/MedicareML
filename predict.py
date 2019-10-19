import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

class PredictDisease():
    def __init__(self, disease):
        model_path = "diabetes.pkl"
        if disease == None:
            model_path = None

        self.model = pickle.load(open(model_path, "rb"))

    def probability(self, features):
        features = np.array([features])
        return self.model.predict_proba(features)

    def goToDoc(self, features, threshold = 0.5):
        return self.probability(features)[0][1] > threshold

def main():
    p = PredictDisease("diabetes")
    print(p.goToDoc([1,2,3,4,5,6,7,8]))



if __name__ == "__main__":
    main()