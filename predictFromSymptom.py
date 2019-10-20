import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier

class PredictDiseaseSymptom():
    def __init__(self, model_path):
        self.model = pickle.load(open(model_path, "rb"))

    def predict(self, symptoms):
        symptom_list = ['itching', 'skin_rash', 'nodal_skin_eruptions',
       'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
       'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
       'vomiting', 'burning_micturition', 'spotting_ urination',
       'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
       'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
       'patches_in_throat', 'irregular_sugar_level', 'cough',
       'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
       'dehydration', 'indigestion', 'headache', 'yellowish_skin',
       'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
       'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
       'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
       'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
       'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
       'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
       'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
       'fast_heart_rate', 'pain_during_bowel_movements',
       'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
       'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
       'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
       'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
       'excessive_hunger', 'extra_marital_contacts',
       'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
       'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'movement_stiffness', 'spinning_movements',
       'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
       'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
       'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
       'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
       'altered_sensorium', 'red_spots_over_body', 'belly_pain',
       'abnormal_menstruation', 'dischromic _patches',
       'watering_from_eyes', 'increased_appetite', 'polyuria',
       'family_history', 'mucoid_sputum', 'rusty_sputum',
       'lack_of_concentration', 'visual_disturbances',
       'receiving_blood_transfusion', 'receiving_unsterile_injections',
       'coma', 'stomach_bleeding', 'distention_of_abdomen',
       'ches', 'fluid_overload.1',
       'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
       'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
       'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
       'inflammatory_nails', 'blister', 'red_sore_around_nose',
       'yellow_crust_ooze']
        feature_vector = np.zeros(len(symptom_list))
        for symptom in symptoms:
            if symptom in symptom_list:
                feature_vector[symptom_list.index(symptom)] = 1
        #print(feature_vector)
        test_dictionary = {
            'Fungal infection':"Visit Dermatologist", 'Allergy':"Visit Dermatologist", 'GERD':"Avoid lying down two hours after sleeping", 'Chronic cholestasis':"Visit urologist",
             'Drug Reaction':"Visit physician immediately", 'Peptic ulcer diseae': "Take antacid", 'AIDS':"Visit physician", 'Diabetes ': "Take glucose test",
             'Gastroenteritis':"Take antacid", 'Bronchial Asthma':"Visit pulmonologist", 'Hypertension ': "Visit cardiologist", 'Migraine': "Visit physician",
             'Cervical spondylosis':"Visit physician", 'Paralysis (brain hemorrhage)':"Visit neurologist", 'Jaundice':"Take bilurubin blood test",
             'Malaria':"Take blood smear test", 'Chicken pox':"Visit physician and remain isolated", 'Dengue':"Take PCR test", 'Typhoid':"Visit physician", 'hepatitis A':"Visit physician",
             'Hepatitis B':"Visit physician", 'Hepatitis C':"Visit physician", 'Hepatitis D':"Visit physician", 'Hepatitis E':"Visit physician",
             'Alcoholic hepatitis':"Visit physician", 'Tuberculosis':"TB test recommended", 'Common Cold':"Self treatable", 'Pneumonia':"Visit physician",
             'Dimorphic hemmorhoids(piles)':"Visit physician", 'Heart attack':"Visit cardiologist immediately", 'Varicose veins': "No apparent disease, visit a physician if symptoms get worse",
             'Hypothyroidism':"Visit physician", 'Hyperthyroidism':"Visit physician", 'Hypoglycemia':"Visit physician",
             'Osteoarthristis':"Visit orthopaedist", 'Arthritis':"Visit orthopaedist",
             '(vertigo) Paroymsal  Positional Vertigo':"Visit orthopaedist", 'Acne':"Visit Dermatologist",
             'Urinary tract infection': "Visit urologist", 'Psoriasis':"Visit physician", 'Impetigo': "Visit physician"

        }
        feature_vector = np.array([feature_vector])
        if np.max(self.model.predict_proba(feature_vector)) > 0.9:
            return test_dictionary[self.model.predict(feature_vector)[0]]
        return "No apparent disease, visit a physician if symptoms get worse"

def main():
    temp = PredictDiseaseSymptom("disease.pkl")
    print(temp.predict(sys.argv[1:]))


if __name__ == "__main__":
    main()