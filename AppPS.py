import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Load the trained machine learning models
model_primary = pickle.load(open('RasaS.pkl', 'rb'))
model_secondary = pickle.load(open('RasaP.pkl', 'rb'))
df = pd.read_csv("dataPS.csv")

def main():
    st.title('Ayurvedic Herbs')

    # Section for input features
    input_col1, input_col2 = st.columns(2)
    # Assuming 'input_features' is defined somewhere in your code
    input_features = [
        "Glucose", "Sucrose", "Fructose", "Tannins",
        "Phenolic Acids", "Citric", "Malic",
        "Tartaric Acid", "Alkaloids", "Terpenes"
    ]

    user_input = {}
    for idx, feature in enumerate(input_features):
        # Splitting features into two columns
        if idx < 5:
            user_input[feature] = input_col1.number_input(f"Enter value for {feature}:", min_value=0.0, max_value=100.0)
        else:
            user_input[feature] = input_col2.number_input(f"Enter value for {feature}:", min_value=0.0, max_value=100.0)

    input_df = pd.DataFrame([user_input])

    if st.button('Predict'):
        prediction_primary = model_primary.predict(input_df)[0]
        prediction_secondary = model_secondary.predict(input_df)[0]

        labels_primary = {
            1: 'Madhura(Sweet)', 2: 'Katu(Pungent)',
            3: 'Kashaya(Astringent)', 4: 'Amla(Sour)',
            5: 'Tikta(Bitter)'
        }

        labels_secondary = {
            1: 'Madhura(Sweet)', 2: 'Katu(Pungent)',
            3: 'Kashaya(Astringent)', 4: 'Amla(Sour)',
            5: 'Tikta(Bitter)'
        }

        predicted_label_primary = labels_primary.get(prediction_secondary, 'Unknown')
        predicted_label_secondary = labels_secondary.get(prediction_primary, 'Unknown')

        st.header('Prediction Results')
        st.write(f"Predicted Primary Rasa: {predicted_label_secondary}")
        st.write(f"Predicted Secondary Rasa: {predicted_label_primary}")

        st.header('Accuracy Graph')

        # Replace with actual accuracy scores
        accuracy_primary_rasa = 0.85

        accuracy_secondary_rasa = 0.92

        labels = ['Primary Rasa', 'Secondary Rasa']
        accuracies = [accuracy_primary_rasa * 100, accuracy_secondary_rasa * 100]

        fig, ax = plt.subplots()
        bars = ax.bar(labels, accuracies, color=['skyblue', 'lightgreen'])
        plt.ylim(0, 100)
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of Rasa Prediction')

        st.pyplot(fig)

if __name__ == '__main__':
    main()