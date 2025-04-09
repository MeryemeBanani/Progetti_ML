import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modello_base import ModelloBase

class ModelloIris(ModelloBase):
    def __init__(self, dataset_path):
        self.dataframe = pd.read_csv(dataset_path)
        self.scaler = None
        self.modello_classificazione = None
        self.classificazione()

    def classificazione(self):
        # Suddivisione del dataset
        y = self.dataframe["class"]
        x = self.dataframe.drop(columns=["class"])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=43)

        # Normalizzazione
        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)

        # Creazione e addestramento del modello
        self.modello_classificazione = Perceptron()
        self.modello_classificazione.fit(x_train, y_train)
        predizioni = self.modello_classificazione.predict(x_test)

        # Valutazione
        print("***** VALUTAZIONE MODELLO CLASSIFICAZIONE *****")
        print(f"L'accuratezza delle predizioni del modello è pari a: {accuracy_score(y_test, predizioni)}")
        print("Il report è:", classification_report(y_test, predizioni), sep="\n")

        # Matrice di confusione
        matrice_predizioni = confusion_matrix(y_test, predizioni)
        print("**** MATRICE DI CONFUSIONE ****", matrice_predizioni, sep="\n")

        # Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrice_predizioni, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.modello_classificazione.classes_,
                    yticklabels=self.modello_classificazione.classes_,
                    cbar=False)
        plt.title("Matrice di Confusione")
        plt.xlabel("Classi Predette")
        plt.ylabel("Classi Reali")
        plt.show()

        # Grafico predizioni
        self.grafico_predizioni(matrice_predizioni)

    @staticmethod
    def grafico_predizioni(matrice_predizioni):
        plt.bar([0, 1.4, 2.8], matrice_predizioni[:, 0], width=0.4, label="setosa", color="blue")
        plt.bar([0.4, 1.8, 3.2], matrice_predizioni[:, 1], width=0.4, label="versicolor", color="green")
        plt.bar([0.8, 2.2, 3.6], matrice_predizioni[:, 2], width=0.4, label="virginica", color="red")
        plt.xticks([0, 1.8, 3.6], ["setosa", "versicolor", "virginica"])
        plt.xlabel("Specie")
        plt.ylabel("Numero di Predizioni")
        plt.legend()
        plt.title("Distribuzione Predizioni")
        plt.show()

    def esportazione(self):
        joblib.dump(self.modello_classificazione, "../modelli/modello_iris.joblib")
        joblib.dump(self.scaler, "../modelli/scaler_iris.joblib")

# Esecuzione
modello = ModelloIris("../dataset/data_06.csv")

# Analisi opzionali:
# modello.analisi_generali(modello.dataframe)
# modello.analisi_valori_univoci(modello.dataframe, ["sepal_length", "sepal_width", "petal_length", "petal_width"])
# modello.analisi_indici_statistici(modello.dataframe)
# modello.individuazione_outliers(modello.dataframe, ["class"])
