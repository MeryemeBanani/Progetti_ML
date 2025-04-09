import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

from machine_learning.analisi.modello_base import ModelloBase

class ModelloHeart(ModelloBase):

    def __init__(self, dataset_path):
        self.dataframe = pd.read_csv(dataset_path)
        self.dataframe_sistemato = self.sistemazione_dataframe()
        self.pca, self.dataframe_ridotto = self.riduzione_dataframe()

    def sistemazione_dataframe(self):
        variabili_quantitative = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
        variabili_categoriali = ["sex"]

        # Standardizzazione delle variabili quantitative
        scaler = StandardScaler()
        df_quantitative = pd.DataFrame(
            scaler.fit_transform(self.dataframe[variabili_quantitative]),
            columns=variabili_quantitative
        )

        # Codifica delle variabili categoriche
        encoder = OneHotEncoder(sparse=False)

        df_categoriali = pd.DataFrame(
            encoder.fit_transform(self.dataframe[variabili_categoriali]),
            columns=encoder.get_feature_names_out(variabili_categoriali)
        )

        # Unione dei due dataframe
        df_sistemato = pd.concat([df_quantitative, df_categoriali], axis=1)
        return df_sistemato

    def riduzione_dataframe(self):
        # Seleziono solo le colonne 'age' e 'trestbps'
        colonne_selezionate = ["age", "chol"]
        dati_filtrati = self.dataframe_sistemato[colonne_selezionate]

        # Applica PCA
        pca = PCA(n_components=2)
        df_ridotto = pd.DataFrame(
            pca.fit_transform(dati_filtrati),
            columns=['age', 'chol']
        )
        return pca, df_ridotto

    def grafico_dispersione_pca(self):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.dataframe_ridotto['age'], self.dataframe_ridotto['chol'], alpha=0.5)
        plt.xlabel('age')
        plt.ylabel('chol')
        plt.title('Grafico di dispersione delle Componenti Principali')
        plt.grid(True)
        plt.show()

# Utilizzo modello
modello = ModelloHeart("../dataset/data_07.csv")
modello.analisi_generali(modello.dataframe_ridotto)
modello.analisi_valori_univoci(modello.dataframe, ["age", "trestbps", "chol", "thalach", "oldpeak"])
modello.analisi_indici_statistici(modello.dataframe)
modello.individuazione_outliers(modello.dataframe, ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"])

print("dataframe sistemato:")
modello.grafico_dispersione_pca()
print(modello.dataframe_sistemato.head())
