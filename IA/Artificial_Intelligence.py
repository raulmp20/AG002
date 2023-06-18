from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


class Inteligencia (object):
    tree = DecisionTreeClassifier()

    def __init__(self, dados):
        self.dados = dados
        # Modificando classe para ficar 1 e 0
        transformacao = {1: 0, 2: 1}
        self.dados['class'] = self.dados[['class']].replace(transformacao)

        dadosX = self.dados.iloc[:, 1:-1]
        dadosY = self.dados.iloc[:, -1]

        print(dadosX)
        print("Começa Y")
        print(dadosY)

        X_train, X_test, y_train, y_test = train_test_split(
            dadosX, dadosY, test_size=0.2, random_state=0)

        self.tree.fit(X_train, y_train)

        y_pred = self.tree.predict(X_test)
        self.acuracia = accuracy_score(y_test, y_pred)

        # extra
        self.matriz_confusao = confusion_matrix(y_test, y_pred)

    def metricas_avaliacao(self, matriz_confusao, acc):
        matriz_confusao = self.matriz_confusao
        acc = self.acuracia
        # IMPRIMINDO A ACURACIA DO TREINAMENTO
        print(f'Acurácia final: {acc*100:.2f}%')

        #### Gráfico da matriz de confusão ####
        print('Matriz de confusão: \n', matriz_confusao)
        # Criar gráfico para matriz de confusão
        plt.figure(figsize=(10, 10))
        plt.imshow(matriz_confusao, interpolation='nearest')
        plt.title('Matriz de confusão')
        # Setar os valores na matriz de confusão
        plt.xticks(range(2), ['O', 'X'])
        plt.yticks(range(2), ['O', 'X'])
        # Setar o valor da matriz de confusão no gráfico
        for i in range(2):
            for j in range(2):
                plt.text(
                    j, i, matriz_confusao[i, j], ha='center', va='center', color='red', fontsize=20)

        plt.colorbar()
        plt.show()
        plt.savefig('matriz_confusao.png')

        plt.close()

    def resultados_treinos(self):
        self.metricas_avaliacao(self.matriz_confusao, self.acuracia)