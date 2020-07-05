import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import set_printoptions
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('fut_bin20_players.csv', delimiter=',')
data.dataframeName = 'Fifa 20 jogadores'

#substitui valores a null
data = data.replace(np.nan, 0)

#print(data.info())

#remover colunas não necessárias
data.drop(columns=['futbin_id', 'player_name', 'player_extended_name', 'quality', 'revision', 'origin', 'club', 'league', 'nationality','date_of_birth', 'intl_rep', 'added_date', 'cb','rb','lb','rwb','lwb','cdm','cm','rm','lm','cam','cf','rf','lf','rw','lw','st','traits','specialities','base_id','resource_id', 'ps4_last', 'ps4_min', 'ps4_max', 'ps4_prp', 'xbox_last','xbox_min','xbox_max','xbox_prp','pc_last','pc_min','pc_max','pc_prp'],inplace=True)

#dicionario das posicoes
pos_dict = {
    'Goalkeeper': ['GK'],
    'Defender': ['LWB', 'RWB', 'LB',  'CB', 'RB'],
    'Midfielder': ['CAM', 'LM', 'CM', 'RM', 'CDM'],
    'Attacker': ['ST', 'LW', 'LF', 'CF', 'RF', 'RW']
}

def simplify_position(position):
    for key in pos_dict:
        if position in pos_dict[key]:
            return key
        else:
            continue
data['SimplifiedPosition']= data['position'].apply(simplify_position)

df_noGK = data[data['SimplifiedPosition'] != 'Goalkeeper']

#Jogadores por posicao
values = data['SimplifiedPosition'].value_counts().values
labels = data['SimplifiedPosition'].value_counts().index.values
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, startangle=90, autopct='%1.2f%%')
#plt.show()


#Atributos jogador baseado na posicao
#df_skills = data.groupby(by='SimplifiedPosition')['dribbling', 'shooting', 'passing', 'defending', 'physicality'].mean().reset_index()

# Correlation matrix
def plotCorrelationMatrix(df):
    #filename = df.dataframeName

    df = df[[col for col in df if df[col].nunique() > 1]]
    # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(14, 12), facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for fifa 20 players', fontsize=15)
    plt.show()

#plotCorrelationMatrix(data) #correlation matrix


#Clustermap - Skills by position
df_skills = data.groupby(data['SimplifiedPosition']).apply(np.mean).T
#print("\n\nSkills by Position:\n",df_skills)
heat_map = sns.clustermap(df_skills, metric='correlation', method='single', cmap='RdBu', standard_scale=0, annot=True, fmt = '.1f', annot_kws={"size": 8})
#plt.show()

#Heatmap - Skills by position
plt.figure(figsize = (15,15))
plt.title('Overall Mean Attributes by position', fontsize=14, fontweight='bold')
sns.heatmap(df_skills, cmap='RdBu',
            vmin=0, vmax=100,
            fmt = '.2f',
            annot=True, annot_kws={"size": 10}
           )
#plt.show()


###    Classification (baseado na Posicao)
df_pos = data.copy().drop(columns=['position','att_workrate', 'def_workrate','pref_foot', 'weak_foot', 'skill_moves'])
X = df_pos.drop(columns=['SimplifiedPosition'])
X = pd.get_dummies(X)
Y = data['SimplifiedPosition']

# Separa dataset entre sets de train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#Selecao dos algoritmos
"""
models = []
models.append(('LR', LogisticRegression(solver ='lbfgs', max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))

# evaluate each model in turn
results = []
names = []
print("\n\nAlgoritmos (Classification):")
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=False)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
    print(msg)

#Selecionar Melhor Modelo / Comparar Algoritmos
fig = plt.figure()
fig.suptitle("Comparação de Algoritmos")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()
"""
#Fazer previsoes na validacao do dataset
LogRegModel = LogisticRegression()
LogRegModel.fit(X_train, Y_train)
prediction = LogRegModel.predict(X_test)

print("\n",classification_report(Y_test, prediction))
print("\n",confusion_matrix(Y_test, prediction))
classifAccuracy = accuracy_score(Y_test, prediction)
print('\nAccuracy Score: ', classifAccuracy)


def pos_numeric(val):
    if val == 'Goalkeeper':
        return 0
    elif val == 'Defender':
        return 1
    elif val == 'Midfielder':
        return 2
    else:
        return 3

df_pos['SimplifiedPosition'] = df_pos['SimplifiedPosition'].apply(pos_numeric)

print("\n\nCorrelacao com posicao:\n", df_pos.corr().abs()['SimplifiedPosition'].sort_values(ascending=False))


###     Regression (prever Overall do dataset)
df_ovr = df_pos.copy()
X = df_ovr.drop(columns='overall')
X = pd.get_dummies(X)
Y = df_ovr['overall']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

"""
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('RFR',RandomForestRegressor(n_estimators=100, random_state=0)))

results = []
names = []
print("\n\nAlgoritmos (Regression):")
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=False)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Selecionar Melhor Modelo / Comparar Algoritmos
fig = plt.figure()
fig.suptitle("Comparação de Algoritmos")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()
"""
#Fazer previsoes na validacao do dataset
RFR = RandomForestRegressor(n_estimators=100, random_state=0)
RFR.fit(X_train, Y_train)
prediction = RFR.predict(X_test)

print ('\n\nMAE: {}'.format(mean_absolute_error(Y_test,prediction)))
print ('R2 score: {}'.format(r2_score(Y_test, prediction)))

print("\n\n",df_ovr.corr().abs()['overall'].sort_values(ascending=False))

# plot actual vs. predicted values
plt.rcParams.update({'font.size': 12})
plt.title('Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.scatter(Y_test, prediction)
plt.plot(Y_test,prediction, color = 'r')
plt.show()


#########################################################
print("\n\n_______________CONCLUSAO_______________")
print("Com {:.2f}% accuracy conseguimos prever a posicao do jogador apenas sabendo os atributos".format(classifAccuracy*100))
print("Para aumentar a performance overall estes atributos sao os que influenciam mais:")
print(df_ovr.corr()['overall'].sort_values(ascending=False).head(10))

print("\n\nA Tabela que é apresentada a seguir mostranos os coeficientes:")
#coef = pd.DataFrame(data = RFR.feature_importances_, index = X_train.columns, columns = ['Coef'])
#print(coef)
print("Se nos fixarmos as outras features e aumentarmos 'drib_reactions' em uma unidade, iremos obter um aumento no 'overall' cerca de 0.265092")


#print(RFR.feature_importances_)
"""
print(df_ovr.head(10))
importance = RFR.feature_importances_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
"""
# importancia features


def convertTuple(tup):
    str = ''.join(tup)
    return str

atributos = []
for i in df_ovr.drop(columns=['overall']):
    atributos.append(i)
#print(pd.DataFrame({'Skills':atributos, 'Importance':RFR.feature_importances_}).sort_values('Importance', ascending=False))

def rf_feat_importance(m):
    return pd.DataFrame({'Skills':atributos, 'Importance':m.feature_importances_}
                       ).sort_values('Importance', ascending=False)

fi = rf_feat_importance(RFR)
var = fi[:10] # top 10
#print(fi)

def plot_fi(fi):
    return fi.plot('Skills', 'Importance', 'barh', figsize=(12,7), legend=False, color = 'b')

# top 10 - atributos para prever overall
plot_fi(fi.nlargest(10, 'Importance'))
plt.show()
