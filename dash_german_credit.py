import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import pickle

df = pd.read_csv('C:\pythons\ger_credit1.csv', index_col=0)

df.head()
X = df.drop('Risk', axis=1)
y = df['Risk']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import  numpy as np
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=22, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
sm = SMOTE(random_state=12)
X_train, y_train = sm.fit_sample(X_train, y_train)

param_grid = [{'max_depth': np.arange(3, 15),
              'min_samples_leaf': [1,3, 5, 10, 20, 30],
              'n_estimators': [10, 50, 100]}]

# build a model
forest = RandomForestClassifier(criterion='gini',
                                    bootstrap=True,
                                    random_state=30,
                                    max_features='sqrt')

folds = 5
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
# build a Grid Search
gs = GridSearchCV(forest, param_grid=param_grid, cv=skf.split(X_train,y_train), scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)

# get the best estimator
forest_clf = gs.best_estimator_

# calculate predictions and probabilities
y_pred = forest_clf.predict(X_test)
y_prob = forest_clf.predict_proba(X_test)[:, 1]

# print some basic information about the model
print('-' * 30)
print('Best parameters:\n\t', gs.best_params_)
print(f'Accuracy score: {round(forest_clf.score(X_test, y_test) * 100, 4)} %')
print(f'AUC {roc_auc_score(y_test, y_prob)}')
print('-' * 30)
print('Classification report:\n', classification_report(y_test, y_pred))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H3('Model uczenia maszynowego - Prognozowanie zdolności kredytowej'),
        html.H6('Model lasów losowych')
    ], style={'textAlign': 'center'}),
    html.Hr(),
    html.Div([
        html.Label('Określ czas trwania kredytu w miesiącach:'),
        dcc.Slider(
            id='slider-1',
            min= 0,
            max=100,
            step=1,
            marks={i : str(i) for i in range(0, 101, 5)},
            tooltip={'placement': 'bottom'}
        ),
        html.Hr(),
        html.Label('Podaj wartość kredytu:'),
        dcc.Input(
            id='input-1',
            type='number'

        ),
        html.Hr(),
        html.Label('Podaj swój wiek:'),
        dcc.Slider(
            id='slider-2',
            min= 18,
            max=90,
            step=1,
            marks={i : str(i) for i in range(18, 91, 3)},
            tooltip={'placement': 'bottom'}
        ),
        html.Br(),
        html.Label('Płeć:'),
        html.Div([
            dcc.RadioItems(
                id='radioitems-1',
                options=[{'label': i, 'value': j} for i,j in zip(['mężczyzna','kobieta'],[1,0])]
            )
        ],style={'width': '20%', 'textAlign':'left'}),
        html.Br(),
        html.Label('Zawód:'),
        html.Div([
            dcc.Dropdown(
                id='dropdown-2',
                options=[{'label': i, 'value': j} for i,j in zip(['niewykwalifkowany','wykwalifikowany','wysko wykwalifikowany'],[1,2,3])]
            )
        ],style={'width': '35%', 'textAlign':'left'}),
        html.Br(),
        html.Label('Mieszkanie:'),
        html.Div([
            dcc.Dropdown(
                id='dropdown-3',
                options=[{'label': i, 'value': j} for i,j in zip(['własne','wynajmowane','socjalne'],['own','rent','free'])]
            )
        ],style={'width': '35%', 'textAlign':'left'}),
        html.Br(),
        html.Label('Środki na koncie oszczędnościowym:'),
        html.Div([
            dcc.Dropdown(
                id='dropdown-4',
                options=[{'label': i, 'value': j} for i, j in
                         zip(['brak', 'mało', 'umiarkowanie','dużo', 'bardzo dużo'], ['lack', 'little', 'moderate','quite rich', 'rich'])]
            )
        ], style={'width': '35%', 'textAlign': 'left'}),
        html.Br(),
        html.Label('Środki na rachunku bieżącym:'),
        html.Div([
            dcc.Dropdown(
                id='dropdown-5',
                options=[{'label': i, 'value': j} for i, j in
                         zip(['brak', 'mało', 'umiarkowanie','dużo'], ['lack', 'little', 'moderate','rich'])]
            )
        ], style={'width': '35%', 'textAlign': 'left'}),
        html.Br(),
        html.Label('Cel kredytu:'),
        html.Div([
            dcc.Dropdown(
                id='dropdown-6',
                options=[{'label': i, 'value': j} for i, j in
                         zip(['radio/TV', 'edukacja', 'meble/wyposażenie', 'samochód','biznes','urządzenia domowe', 'naprawy', 'wczasy'], ['radio/TV', 'education', 'furniture/equipment', 'car', 'business','domestic appliances', 'repairs', 'vacation/others'])]
            )
        ], style={'width': '35%', 'textAlign': 'left'}),
        html.Div([
            html.Hr(),
            html.H3('Ocena zdolności kredytowej'),
            html.Hr(),
            html.H4('Wypełniony formularz kredytowy:'),
            html.Div(id='div-1'),
            html.Div(id='div-2'),
            html.Hr()

        ], style={'margin':'0 auto', 'textAlign':'center'})

    ], style={'width': '90%', 'textAlign':'left', 'margin':'0 auto', 'fontSize':22})
])

sex = {1:'mężczyzna', 0:'kobieta'}
job = {1:'niewykwalifikowany', 2:'wykwalifikowany', 3:'wysoko wykwalifikowany'}
housing = {'own': 'włąsne', 'rent':'wynajmowane', 'free':'socjalne'}
savings = {'lack':'brak', 'little':'mało', 'moderate':'umiarkowanie','quite rich':'dużo', 'rich':'bardzo dużo'}
current = {'lack':'brak', 'little':'mało', 'moderate':'umiarkowanie','rich':'dużo'}
purpose = {'radio/TV':'radio/TV', 'education':'edukacja', 'furniture/equipment':'meble/wyposażenie', 'car':'samochód', 'business':'biznes','domestic appliances': 'urządzenia domowe', 'repairs':'naprawy', 'vacation/others':'wczasy'}
@app.callback(
    Output(component_id='div-1', component_property='children'),
    [Input('slider-1','value'),
     Input('input-1','value'),
     Input('slider-2','value'),
     Input('radioitems-1','value'),
     Input('dropdown-2','value'),
     Input('dropdown-3','value'),
     Input('dropdown-4','value'),
     Input('dropdown-5','value'),
     Input('dropdown-6','value')]
)
def display_parametrs(val1,val2,val3,val4,val5,val6,val7,val8,val9):
    if val1 and val2 and val3 and val4 and val5 and val6 and val7 and val8 and val9:
        val4 = sex[val4]
        val5 = job[val5]
        val6 = housing[val6]
        val7 = savings[val7]
        val8 = current[val8]
        val9 = purpose[val9]
        return html.Div([
            html.H6(f'Czas trwania kredytu: {val1} miesięcy'),
            html.H6(f'Wartość kredytu: {val2} zł'),
            html.H6(f'Wiek kredytobiorcy: {val3} lat'),
            html.H6(f'Płeć: {val4}'),
            html.H6(f'Zawód: {val5}'),
            html.H6(f'Mieszkanie: {val6}'),
            html.H6(f'Środki na koncie oszczędnościowym: {val7}'),
            html.H6(f'Środki na koncie bieżącym: {val8} '),
            html.H6(f'Cel kredytu: {val9}')
        ], style={'textAlign':'left'})
    else:
        return html.Div([
            html.H6('Wypełnij wszystkie pola')
        ])
@app.callback(
    Output(component_id='div-2', component_property='children'),
    [Input('slider-1','value'),
     Input('input-1','value'),
     Input('slider-2','value'),
     Input('radioitems-1','value'),
     Input('dropdown-2','value'),
     Input('dropdown-3','value'),
     Input('dropdown-4','value'),
     Input('dropdown-5','value'),
     Input('dropdown-6','value')]
)
def display_parametrs1(val1,val2,val3,val4,val5,val6,val7,val8,val9):
    if val1 and val2 and val3 and val4 and val5 and val6 and val7 and val8 and val9:
       val6_1, val6_2, val6_3 = 0, 0, 0
       val7_1, val7_2, val7_3, val7_4, val7_5 = 0, 0, 0, 0, 0
       val8_1, val8_2, val8_3, val8_4 = 0, 0, 0, 0
       val9_1, val9_2, val9_3, val9_4, val9_5, val9_6, val9_7, val9_8 = 0, 0, 0, 0, 0, 0, 0, 0
       if val6 == 'free':
           val6_1=1
       elif val6 == 'own':
            val6_2=1
       elif val6 == 'rent':
            val6_3=1

       if val7 == 'lack':
            val7_1=1
       elif val7 == 'little':
           val7_2=1
       elif val7== 'moderate':
           val7_3 = 1
       elif val7 == 'quite rich':
           val7_4 =1
       elif val7 == 'rich':
           val7_5=1


       if val8 == 'lack':
           val8_1 = 1
       elif val8 == 'little':
           val8_2 = 1
       elif val8 == 'moderate':
           val8_3 = 1
       elif val8 == 'rich':
           val8_4 = 1

       if val9 == 'business':
            val9_1=1
       elif val9 =='car':
            val9_2=1
       elif val9 == 'domestic appliances':
            val9_3=1
       elif val9 == 'education':
            val9_4=1
       elif val9 == 'furniture/equipment':
            val9_5=1
       elif val9 == 'radio/TV':
            val9_6=1
       elif val9 == 'repairs':
            val9_7=1
       elif val9 == 'vacation/others':
            val9_8=1
       df_sample= pd.DataFrame(
            data=[[val1,val2,val3,val4,val5,val6_1, val6_2, val6_3,val7_1, val7_2, val7_3, val7_4, val7_5,val8_1, val8_2, val8_3, val8_4,val9_1, val9_2, val9_3, val9_4,val9_5, val9_6, val9_7, val9_8]],
            columns=['Duration','Credit amount', 'Age','Sex', 'Job','Housing_free','Housing_own','Housing_rent',
                    'Saving accounts_lack','Saving accounts_little','Saving accounts_moderate','Saving accounts_quite rich',
                    'Saving accounts_rich','Checking account_lack','Checking account_little','Checking account_moderate',
                    'Checking account_rich','Purpose_business','Purpose_car','Purpose_domestic appliances',
                    'Purpose_education','Purpose_furniture/equipment','Purpose_radio/TV','Purpose_repairs','Purpose_vacation/others']


        )
       pd.set_option('display.max_columns', 500)
       print(df_sample)
       credit = forest_clf.predict(df_sample)[0]

       def change(cred):
           if cred == 1:
               print('pozytywna')
           else:
               print('negatywna')
       credi = change(credit)



       return html.Div([
           html.H4(f'Ocena credit scoringu: {credit}')
       ])

   # print(val1,val2,val3,val4,val5,val6,val7,val8,val9)
if __name__ == '__main__':
    app.run_server(debug=True)

import xgboost
