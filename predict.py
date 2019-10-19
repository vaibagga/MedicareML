import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

class PredictDisease():
    def __init__(self, disease):
        if disease not in ["cardio", "diabetes", "liver"]:
            print("Not a disease")
        model_path = disease + ".pkl"
        self.model = pickle.load(open(model_path, "rb"))

    def probability(self, features):
        features = np.array([features])
        return self.model.predict_proba(features)

    def goToDoc(self, features, threshold = 0.5):
        """

        :param features: Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio in liver
        Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age in diabetes
        id,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active in cardio
        :param threshold: the threshold probability above which the patient should go to doctor
        :return: whether the patient should consult doctor
        """
        return self.probability(features)[0][1] > threshold

def main():
    p = PredictDisease("diabetes")
    print(p.goToDoc([1,2,3,4,5,6,7,8]))



if __name__ == "__main__":
    main()