import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def DataAnalysis(data):

    # Quants atributs té la vostra base de dades?
    print("Dimensionalitat de la BBDD:", data.shape)
    print("Número de instancies:", data.shape[0])
    print("Número de variables:", data.shape[1])

    # Quin tipus d'atributs tens? (Númerics, temporals, categorics, binaris...)
    print(data.info())

    print("Podem veure que el dataset es compon de un mix entre variables categòriques i númeriques.")
    print("Les variables categòriques tenen el tipus de dades Object")
    print("Les variables númeriques tenen el tipus de dades Float64")

    # Com es el target, quantes categories diferents existeixen?
    print("El target es tracta de la variable: RainTomorrow")
    print("Número de nulls de RainTomorrow:", data['RainTomorrow'].isnull().sum())
    print("Valors i tipus de RainTomorrow:", data['RainTomorrow'].unique())
    print("Distribució de freqüència dels valors de RainTomorrow:", data['RainTomorrow'].value_counts())
    print("Existeixen diferents categroies, per exemple: ") #Revisar Categories

    # Podeu veure alguna correlació entre X i y?
    data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})
    data['RainToday'] = data['RainToday'].map({'Yes': 1, 'No': 0})



    print(data)

    fig, ax = plt.subplots(1, 2)


    plt.figure(figsize=(20, 20))
    sns.countplot(data=data, x='RainToday', ax=ax[0])
    sns.countplot(data=data, x='RainTomorrow', ax=ax[1])
    plt.show()

    #Aquí s'han anat canviant els paramentres MaxTemp i MinTemp
    sns.violinplot(x='RainTomorrow', y='MinTemp', data=data, hue='RainTomorrow')
    plt.show()



    correlation = data.corr()
    plt.figure(figsize=(20, 12))
    plt.title('Correlació de correlació')
    ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)

    plt.savefig("HeatMap.png", bbox_inches="tight")
    plt.show()
    # Estan balancejades les etiquetes (distribució similar entre categories)? Creus que pot afectar a la classificació la seva distribució?
