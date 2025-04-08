from unicodedata import category

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

#one hot encoding
df = pd.read_csv("breast-cancer.csv", header= None)
df.columns = ['age','menopause' , 'tumer-size' ,'inv-nodes' ,	'node-caps' 	,'deg-malig' ,	'breast' ,	'breast-quad' 	,'irradiate', 'class' ]
df = pd.DataFrame(df)

print(df.info())
print(df.shape)

columns = ['age','menopause' , 'tumer-size' ,'inv-nodes' ,	'node-caps'  ,	'breast' ,	'breast-quad'  ]
target_columns =  df[columns].columns
print(target_columns)

print("target columns: ",target_columns)
enc = OneHotEncoder(sparse_output=False)

enc.fit(df[target_columns])
# print(enc.categories_)
encoded = enc.transform(df[target_columns] )
print(df.describe())

encoded_mat = pd.DataFrame(encoded, columns=enc.get_feature_names_out(target_columns))

print(encoded_mat.info())


#ordinal encoding
ord_en= OrdinalEncoder(categories=[["'no'", "'yes'"]])
df['irradiate'] = ord_en.fit_transform(df[['irradiate']])
print(enc.categories_)

#label encoding
lab_enco = LabelEncoder()
df['class'] = lab_enco.fit_transform(df['class'])
df['deg-malig'] = lab_enco.fit_transform(df['deg-malig'])


#Data preparation

df_enc = encoded_mat.join(df['deg-malig'])
df_enc = df_enc.join(df['irradiate'])

print(df_enc.head(10))
X = df_enc
y=df['class']

#Training
print(X.shape, y.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#MODELLING DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=22)

log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
acc_score = accuracy_score(y_test,y_pred)
print("\nAccuracy score: ",acc_score)