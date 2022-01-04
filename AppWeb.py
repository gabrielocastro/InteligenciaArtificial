import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#lendo e exibindo a tabela
st.title("Comparação Messi-Ronaldo por clubes entre as temporadas 2009/10-2017/18")
df = pd.read_csv('MessiRonaldo.csv')
st.text("Este conjunto de dados contém os seguintes recursos:\n"
        "Season: Temporada.\n"
        "Player: Jogador.\n"
        "Liga_Goals: Gols em La Liga.\n"
        "Liga_Asts: Assistências em La Liga.\n"
        "Liga_Aps: Jogos em La Liga.\n"
        "Liga_Mins: Minutos em La Liga.\n"
        "CL_Goals: Gols na Champions League.\n"
        "CL_Asts: Assistências na Champions League.\n"
        "CL_Aps: Jogos na Champions League.\n"
        "CL_Mins: Minutos na Champions League.\n"
        "Dataset retirado do site Kaggle.")
st.subheader(' ')
st.subheader(' ')


st.subheader('Tabela exibindo estatísticas dos jogadores ao longo dos anos:')
df
df.info(verbose=True)
df.describe()


#Análise Exploratória
st.title("Análise exploratória")
#histograma
st.subheader('Histograma exibindo o números jogos disputados por temporada:')
st.set_option('deprecation.showPyplotGlobalUse', False)
ht = pd.DataFrame(df, columns=['Liga_Aps'])
ht.hist()
plt.show()
st.pyplot()
st.subheader(' ')
st.subheader(' ')
#jointplot
with sns.axes_style('white'):
    st.subheader('Gráfico jointplot do tipo scatter Gols na UCL por temporada:')
    sc = sns.jointplot('CL_Goals', 'Season', df, kind='scatter', color="midnightblue")
    st.pyplot(sc)
    st.subheader(' ')
    st.subheader(' ')
    st.subheader('Gráfico jointplot do tipo kde Gols em La Liga por minutos jogados:')
    kd = sns.jointplot("Liga_Goals", "Liga_Mins", df, kind='kde', fill=True, thresh=0, color="darkred")
    st.pyplot(kd)
    st.subheader(' ')
    st.subheader(' ')
    st.subheader('Gráfico jointplot do tipo scatter Gols em La Liga por temporada:')
    sc2 = sns.jointplot("Liga_Goals", "Season", df, kind='scatter', color="goldenrod")
    st.pyplot(sc2)
    st.subheader(' ')
    st.subheader(' ')
#pairplot
st.subheader('Gráfico pairplot partidas jogadas em La Liga:')
pp = pd.DataFrame(df, columns=['Liga_Goals', 'Liga_Asts', 'Liga_Aps', 'Liga_Mins'])
fig = sns.pairplot(pp, hue='Liga_Aps')
st.pyplot(fig)
st.write('Parâmetros utilizados: Minutos, gols, assistências e partidas jogadas em La Liga.')
st.subheader(' ')
st.subheader(' ')


#regressão logística
st.title('Exibição dos resultados da Regressão Logística')
X = df.drop(["Season",	"Player"], axis=1)
y = df["Player"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
#Exibindo os resultados
prediction = logreg.predict(X_test)
st.text('Model Report:\n ' + classification_report(y_test, prediction))

