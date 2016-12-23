# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

train = pd.read_csv("train.csv", parse_dates = ["Dates"], index_col= False)
test = pd.read_csv("test.csv",parse_dates=["Dates"], index_col = False)
train.info()


train = train.drop(["GeoName", "ComponentId", "Region", "Description", "County Code","Site Num","Address State",
                    "County"], axis=1)
test = test.drop(["GeoName", "ComponentId", "Region", "Description"], axis = 1)
train.info()


def datesplit(data):
    data["Year"] = data["Dates"].dt.year
    data["Month"] = data["Dates"].dt.month
    data["Day"] = data["Dates"].dt.day
    data["Hour"] = data["Dates"].dt.hour
    return data

train = datesplit(train)
test = datesplit(test)


enc = LabelEncoder()
train["Category"] = enc.fit_transform(train["Category"])
wnc = LabelEncoder()
train["City"] = wnc.fit_transform(train["City"])
cat_encoder = LabelEncoder()
cat_encoder.fit(train["Category"])
train["IndustryClassification"]= cat_encoder.transform(train["IndustryClassification"])
print(cat_encoder.classes_)
enc = LabelEncoder()
test["Category"]= enc.fit_transform(test["Category"])
wnc = LabelEncoder()
test["City"] = wnc.fit_transform(test["City"])
train["X"] = abs(train["X"])
train["Y"] = abs(train["Y"])
test["X"] = abs(test["X"])
test["Y"]= abs(test["Y"])
print(train.columns)
print(test.columns)


train_columns = list(train.columns[1:8].values)
print(train_columns)
test_columns = list(test.columns[1:8].values)
print(test_columns)


training,validation = train_test_split(train, train_size=.75)
model = MultinomialNB()
model.fit(training[train_columns], training["Category"])
predict = model.predict_proba(validation[train_columns])
predicted = model.predict_proba(test[test_columns])    
log_loss(validation["Category"], predict)


result = pd.DataFrame(predicted,columns=cat_encoder.classes_)
result.to_csv("output.csv", index = True, index_label = "Id")
