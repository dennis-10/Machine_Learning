# Aplicação de alguns modelos de Machine Learning

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Do módulo no qual selecionamos a maneira de ter os dados
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder # Do módulo de pré-processamento do sklearn
from sklearn.metrics import confusion_matrix, accuracy_score # Métricas para entender os resultados do modelo
from sklearn.tree import DecisionTreeClassifier, export_graphviz, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from yellowbrick.classifier import ConfusionMatrix


data = pd.read_csv('netflix_titles.csv', sep = ',') # Dataset relacionado a todos os show e filmes da Netflix

# Removerei a coluna 'descrição' pois não realizarei mineração de texto
data.drop('description', inplace=True, axis=1) # axis = 1 diz q é a Coluna e não uma linha
#  Se já tenho o ID, não preciso do título
data.drop('title', inplace=True, axis = 1)
# Elenco sai pelo mesmo motivo que 'description'
data.drop('cast', inplace=True, axis = 1)
# inplace = True diz que substituirá nos dados originais

# Removendo NAN's
data['director'] = data['director'].replace(np.nan, '0')

# Moda
data['rating'].isna().sum()# 8 na's
data.groupby(['rating']).size().sort_values(ascending=False) # A moda é TV-MA
data['rating'] = data['rating'].replace(np.nan, 'TV-MA')

# EUA é a moda
data.groupby(['country']).size().sort_values(ascending=False) # Pegando os dados por grupo em ordem

# Substituindo pela Moda
data['country'] = data['country'].replace(np.nan, 'United States')

# Algumas linhas se referem a séries/shows, são as que têm duração em "Seasion"
data['duration'].head()
# Pela categoria
consulta = data.type.str.contains('Movie')
data = data[consulta]

# Tiramos 'type' porque se tornou inútil
data.drop('type', inplace=True, axis = 1)


# Irei dizer quais são os filmes que gosto e não gosto criando uma coluna de classificação nos dados de treino
np.random.seed()
data['Classificacao'] = np.random.choice([0,1], size=len(data), p=[0.70,0.30])


# Passando para numérico, pois a coluna 'duration' está como string(categórico)
data.duration = data.duration.str.extract('(\d+)') # Use regex/expressão regular

# Passar as datas para string
data['date_added'] = data['date_added'].astype(str)

# Transforma em matriz os valores do dataframe para realizar a transformação usando LabelEncoder
previsores = data.iloc[:,0:8].values # dtype é 'O' de Object, que indica que esse objeto pandas tem mais de um
# tipo de variável, 'str' e 'float' por exemplo.
classe = data.iloc[:,8]

# Transformando valores categóricos que restaram em numéricos
labelencoder = LabelEncoder()
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,4] = labelencoder.fit_transform(previsores[:,4])
previsores[:,5] = labelencoder.fit_transform(previsores[:,5])
previsores[:,7] = labelencoder.fit_transform(previsores[:,7])

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size=0.3,
                                                                  random_state=0)

# Naive bayes
naive_bayes = GaussianNB()
naive_bayes.fit(x_treinamento,y_treinamento)

previsoes = naive_bayes.predict(x_teste)
print(previsoes)
confusao = confusion_matrix(y_teste,previsoes)
print(confusao)
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto
print(taxa_acerto) #70%


# Árvore de decisão
arvore = DecisionTreeClassifier()
arvore.fit(x_treinamento,y_treinamento)

previsoes = arvore.predict(x_teste)
print('\n\nArvore:')
print(previsoes)
confusao  = confusion_matrix(y_teste, previsoes)
print(confusao)
taxa_acerto = accuracy_score(y_teste, previsoes)
print(taxa_acerto) # Em torno de 57%

# Exportação da árvore de decisão para o formato .dot, para posterior visualização
export_graphviz(arvore, out_file = 'tree.dot') # http://www.webgraphviz.com/


# Máquina de Vetor de Suporte
svm = SVC()
svm.fit(x_treinamento, y_treinamento)

previsoes = svm.predict(x_teste)
print('\n\nMaquina de Vetor de Suporte: ')
print(previsoes)

taxa_acerto= accuracy_score(y_teste, previsoes)
print(taxa_acerto)

# Vizinho Mais Próximo
knn = KNeighborsClassifier(n_neighbors= 3)
knn.fit(x_treinamento,y_treinamento)

previsoes = knn.predict(x_teste)
print('\n\nVizinho mais proximo')
print(previsoes)

confusao = confusion_matrix(y_teste, previsoes)
print(confusao)

taxa_acerto = accuracy_score(y_teste, previsoes)
print(taxa_acerto)

floresta = RandomForestClassifier(n_estimators=100)
floresta.fit(x_treinamento, y_treinamento)

floresta.estimators_[1]


previsoes = floresta.predict(x_teste)
print('\n\nFloresta aleatória: ')
print(previsoes)

confusao = confusion_matrix(y_teste, previsoes)
print(confusao)

taxa_acerto = accuracy_score(y_teste, previsoes)
print(taxa_acerto)