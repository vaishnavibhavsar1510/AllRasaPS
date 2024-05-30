import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataPS.csv")
df.head()

features = df[['Glucose','Sucrose','Fructose','Tannins','Phenolic Acids','Citric','Malic','Tartaric Acid','Alkaloids','Terpenes']]

primary_rasa = df['Primary']
secondary_rasa = df['Secondary']

features_train, features_test, primary_rasa_train, primary_rasa_test, secondary_rasa_train, secondary_rasa_test = train_test_split(
    features, primary_rasa, secondary_rasa, test_size=0.2, random_state=42
)

# Create KNN classifiers for primary and secondary rasas
clf_primary_rasa = KNeighborsClassifier(n_neighbors=5)
clf_secondary_rasa = KNeighborsClassifier(n_neighbors=5)
clf_primary_rasa.fit(features_train, primary_rasa_train)
clf_secondary_rasa.fit(features_train,secondary_rasa_train)

import pickle

import pickle

# Assuming 'clf' is your trained model
with open(r'C:\Users\Vaishnavi\OneDrive\Desktop\AllRasaPS\RasaS.pkl', 'wb') as file:
    pickle.dump(clf_primary_rasa, file)
with open(r'C:\Users\Vaishnavi\OneDrive\Desktop\AllRasaPS\RasaP.pkl', 'wb') as file:
    pickle.dump(clf_secondary_rasa, file)

# Load the pickled model
with open(r'C:\Users\Vaishnavi\OneDrive\Desktop\AllRasaPS\RasaS.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
with open(r'C:\Users\Vaishnavi\OneDrive\Desktop\AllRasaPS\RasaP.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Make predictions on the test set
primary_rasa_pred = clf_primary_rasa.predict(features_test)
secondary_rasa_pred = clf_secondary_rasa.predict(features_test)

# Calculate accuracy scores
accuracy_primary_rasa = accuracy_score(primary_rasa_test, primary_rasa_pred) * 100
accuracy_secondary_rasa = accuracy_score(secondary_rasa_test, secondary_rasa_pred) * 100

print(f'Accuracy for Primary Rasa: {accuracy_primary_rasa:.2f}%')
print(f'Accuracy for Secondary Rasa: {accuracy_secondary_rasa:.2f}%')

# Plot a bar chart for better visualization with attractive colors
labels = ['Primary Rasa', 'Secondary Rasa']
accuracies = [accuracy_primary_rasa, accuracy_secondary_rasa]

colors = ['skyblue', 'lightgreen']

plt.bar(labels, accuracies, color=colors)
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Rasa Prediction')
plt.show()
