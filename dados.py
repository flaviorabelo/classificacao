import pandas as pd
from collections import Counter

df = pd.read_csv('evasao_curso_semestre_treino_MSDOS.csv', sep=';')

X_df = df[['ID_CURSO', 'SEMESTRE', 'CH_CURSADA_TOTAL', 'CH_TOTAL_CURSO', 'CH_TOTAL_SEMESTRE', 'CH_APROV_SEM', 'MEDIA_DISC_SEMESTRE', 'TOT_REPROV', 'QTD_DISC_SEM', 'PERC_REPROV', 'BOM_PAGADOR']]
Y_df = df['CLASSE']

X = X_df
Y = Y_df

porcentagem_treino = 0.8
porcentagem_teste = 0.1

tamanho_de_treino = int(porcentagem_treino * len(Y))
tamanho_de_teste = int(porcentagem_teste * len(Y))
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

print("Tamanho Treino: %d:" % tamanho_de_treino)
print("Tamanho Teste: %d:" % tamanho_de_teste)
print("Tamanho Validacao: %d:" % tamanho_de_validacao)

fim_de_treino = tamanho_de_treino + tamanho_de_teste

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[tamanho_de_treino:fim_de_treino]
teste_marcacoes = Y[tamanho_de_treino:fim_de_treino]

validacao_dados = X[fim_de_treino:]
validacao_marcacoes = Y[fim_de_treino:]

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes,teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)
    resultado = modelo.predict(teste_dados)

    acertos = resultado == teste_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(teste_dados)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do algoritmo {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)

    return taxa_de_acerto

def teste_real(modelo, validacao_dados, validacao_marcacoes):

    resultado = modelo.predict(validacao_dados)

    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)

    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    print(msg)


from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()


# Treino do Modelo
print('Treinando modelo...')
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaboostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

if resultadoMultinomial > resultadoAdaBoost:
    vencedor = modeloMultinomial
else:
    vencedor = modeloAdaBoost

teste_real(vencedor, validacao_dados, validacao_marcacoes)

# a eficácia do algoritmo que chuta tudo 0 ou 1
acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(teste_dados)
print("Total de teste: %d" % total_de_elementos)
