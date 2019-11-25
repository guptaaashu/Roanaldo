# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 01:51:11 2019

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

database=pd.read_csv('data56.csv')
data= database.dropna(subset=['is_goal'])

plt.bar(data.is_goal, data.shot_basics)
plt.xlabel('Genre', fontsize=5)
plt.ylabel('No of Movies', fontsize=5)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Market Share for Each Genre 1995-2017')
plt.show()

var = data[data.is_goal==0].groupby(['shot_basics']).is_goal.count()
var1 = data[data.is_goal==1].groupby(['shot_basics']).is_goal.count()
var2= var1/(var+var1)
var2.plot(kind='bar', grid=False)

var = data[data.is_goal==0].groupby(['power_of_shot.1']).is_goal.count()
var1 = data[data.is_goal==1].groupby(['power_of_shot.1']).is_goal.count()
var2= var1/(var+var1)
var2.plot()

var = data[data.is_goal==0].groupby(['knockout_match.1']).is_goal.count()
var1 = data[data.is_goal==1].groupby(['knockout_match.1']).is_goal.count()
var2= var1/(var+var1)
var1.plot()


var = data[data.is_goal==0].groupby(['game_season']).is_goal.count()
var1 = data[data.is_goal==1].groupby(['game_season']).is_goal.count()
var2= var1/(var+var1)
var2.plot(kind='bar', grid=False)

X = data.iloc[:, [1,7,8,9,11,12,13,19,23,24,25]].values
columns = ['match_event_id', 'game_season','remaining_sec','distance_of_shot','área_of_shot','shot_basics','range_of_shot','type_of_shot','remaining_min','power_shot','knockout']
X=pd.DataFrame(X, columns=columns)

""" Data Prepocessing """

X['match_event_id'].fillna((X['match_event_id'].mean()), inplace=True)
X['match_event_id']= X.match_event_id.astype('int')


X['game_season'].fillna(X['game_season'].mode().values[0], inplace = True)
X['game_season']= X.game_season.astype('category')
X['game_season'] = X["game_season"].cat.codes

X['remaining_sec'].fillna((X['remaining_sec'].mean()), inplace=True)

X['distance_of_shot'].fillna((X['distance_of_shot'].mean()), inplace=True)

X['área_of_shot'].fillna(X['área_of_shot'].mode().values[0], inplace = True)
X['área_of_shot']= X.área_of_shot.astype('category')
X['área_of_shot'] = X["área_of_shot"].cat.codes

X['shot_basics'].fillna(X['shot_basics'].mode().values[0], inplace = True)
X['shot_basics']= X.shot_basics.astype('category')
X['shot_basics'] = X["shot_basics"].cat.codes

X['range_of_shot'].fillna(X['range_of_shot'].mode().values[0], inplace = True)
X['range_of_shot']= X.range_of_shot.astype('category')
X['range_of_shot'] = X["range_of_shot"].cat.codes

X['type_of_shot'].fillna(X['type_of_shot'].mode().values[0], inplace = True)
X['type_of_shot']= X.type_of_shot.astype('category')
X['type_of_shot'] = X["type_of_shot"].cat.codes



X['remaining_min'].fillna((X['remaining_min'].mean()), inplace=True)

X['power_shot'].fillna(X['power_shot'].mode().values[0], inplace=True)

X['knockout'].fillna((X['knockout'].mode().values[0]), inplace=True)

Y= data.iloc[:, [10]].values
columns=['is_goal']
Y=pd.DataFrame(Y, columns=columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.svm import SVC
classif= SVC(kernel='linear',random_state=0)
classif.fit(X_train,y_train)



y_pred=classifier.predict(X_test)
y_pred= (y_pred>0.5)
