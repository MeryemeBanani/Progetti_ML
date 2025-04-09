import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

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

        # Esempio di calcolo di una variabile di rischio su una scala da 0 a 1, sommo i valori relativi dei parametri e normalizzo rispetto al numero di parametri
        df_sistemato['rischio'] = (
                (df_sistemato['chol'] / df_sistemato['chol'].max()) +
                (df_sistemato['trestbps'] / df_sistemato['trestbps'].max()) +
                (df_sistemato['thalach'] / df_sistemato['thalach'].max()) +
                (df_sistemato['age'] / df_sistemato['age'].max()) +
                (df_sistemato['oldpeak'] / df_sistemato['oldpeak'].max())
        )


        df_sistemato['rischio'] = df_sistemato['rischio'] / 5  #normalizzo

        # soglia per classificare i rischi come alto (1) o basso (0) è 0.5
        soglia_rischio = 0.2
        df_sistemato['rischio'] = (df_sistemato['rischio'] > soglia_rischio).astype(int)

        # Ora puoi eseguire la regressione logistica
        x = df_sistemato[["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]]
        y = df_sistemato['rischio']


        model = LogisticRegression()


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Allena il modello
        model.fit(x_train, y_train)

        # Previsioni e valutazione
        y_pred = model.predict(x_test)

        # Misure di performance
        print(f"Accuratezza: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print(classification_report(y_test, y_pred))

        # Area sotto la curva ROC
        y_prob = model.predict_proba(x_test)[:, 1]
        print(f"AUC: {roc_auc_score(y_test, y_prob):.2f}")

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


