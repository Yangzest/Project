
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
import csv


# In[6]:


data_filename = "data/basketball.csv"
dataset = pd.read_csv(data_filename)


# In[7]:


dataset.head()


# In[10]:


dataset = pd.read_csv(data_filename, parse_dates=["Date"])

dataset.columns = ["Date", "Start (ET)", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "OT?", "Score Type", "Attend", "Notes"]


# In[11]:


dataset.head()


# In[12]:


print(dataset.dtypes)


# In[13]:


dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]


# In[14]:


dataset.head()


# In[15]:


y_true = dataset["HomeWin"].values


# In[16]:


#HomeWin percentage
dataset["HomeWin"].mean()


# In[17]:


from collections import defaultdict
won_last = defaultdict(int)


# In[18]:


#whether the team won the last time
dataset["HomeLastWin"] = 0
dataset["VisitorLastWin"] = 0


# In[19]:


# 1 or a 0, depending on which team won the current game
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    dataset.set_value(index, "HomeLastWin", won_last[home_team])
    dataset.set_value(index, "VisitorLastWin", won_last[visitor_team])
    
    won_last[home_team] = int(row["HomeWin"])
    won_last[visitor_team] = 1 - int(row["HomeWin"])


# In[20]:


dataset.head(6)


# In[21]:


dataset.ix[1000:1005]


# In[22]:


#dataset using our last win values for both the home team and the visitor team
X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values


# In[23]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=14)


# In[24]:


from sklearn.cross_validation import cross_val_score
import numpy as np


# In[29]:


scores = cross_val_score(clf, X_previouswins, y_true,
scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


# In[32]:


import os
standings_filename = "data/standings.csv"

standings = pd.read_csv(standings_filename, skiprows=1)


# In[33]:


standings.head()


# In[34]:


dataset["HomeTeamRanksHigher"] = 0
for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    home_rank = standings[standings["Team"] == home_team]["Rk"].values[0]
    visitor_rank = standings[standings["Team"] == visitor_team]["Rk"].values[0]
    dataset.set_value(index, "HomeTeamRanksHigher", int(home_rank < visitor_rank))


# In[35]:


X_homehigher = dataset[[ "HomeTeamRanksHigher", "HomeLastWin", "VisitorLastWin",]].values


# In[39]:


clf = DecisionTreeClassifier(random_state=14, criterion="entropy")

scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')

print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


# In[40]:


last_match_winner = defaultdict(int)
dataset["HomeTeamWonLast"] = 0

for index, row in dataset.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    teams = tuple(sorted([home_team, visitor_team]))  # Sort for a consistent ordering
    # Set in the row, who won the last encounter
    home_team_won_last = 1 if last_match_winner[teams] == row["Home Team"] else 0
    dataset.set_value(index, "HomeTeamWonLast", home_team_won_last)
    # Who won this one?
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
    last_match_winner[teams] = winner


# In[41]:


dataset.ix[400:450]


# In[42]:


X_lastwinner = dataset[[ "HomeTeamWonLast", "HomeTeamRanksHigher", "HomeLastWin", "VisitorLastWin",]].values
clf = DecisionTreeClassifier(random_state=14, criterion="entropy")

scores = cross_val_score(clf, X_lastwinner, y_true, scoring='accuracy')

print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


# In[43]:


from sklearn.preprocessing import LabelEncoder
encoding = LabelEncoder()
encoding.fit(dataset["Home Team"].values)
home_teams = encoding.transform(dataset["Home Team"].values)
visitor_teams = encoding.transform(dataset["Visitor Team"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
X_teams = onehot.fit_transform(X_teams).todense()

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


# In[44]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


# In[45]:


X_all = np.hstack([X_lastwinner, X_teams])
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


# In[46]:


X_all = np.hstack([X_lastwinner, X_teams])
clf = RandomForestClassifier(random_state=14, n_estimators=250)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


# In[49]:


from sklearn.grid_search import GridSearchCV
parameter_space = {
    "max_features": [2, 10, 'auto'],
    "n_estimators": [100, 200],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [2, 4, 6],
}
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))


# In[51]:


print(grid.best_estimator_)

