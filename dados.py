import pandas as pd

df = pd.read_csv('evasao_curso_semestre_treino_MSDOS.csv', sep=';')

X_df = df[['ID_ALUNO', 'ID_CURSO', 'SEMESTRE', 'CH_CURSADA_TOTAL', 'CH_TOTAL_CURSO', 'CH_TOTAL_SEMESTRE', 'CH_APROV_SEM', 'MEDIA_DISC_SEMESTRE', 'TOT_REPROV', 'QTD_DISC_SEM', 'PERC_REPROV', 'BOM_PAGADOR']]
Y_df = df['CLASSE']

X = X_df
Y = Y_df

porcentagem_treino = 0.9

tamanho_de_treino = int(porcentagem_treino * len(Y))
tamanho_de_teste = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()

# Treino do Modelo
print('Treinando modelo...')
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * (total_de_acertos / total_de_elementos)

print(taxa_de_acerto)
print(total_de_elementos)
