import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#Data Preprocessing

train_pd=pd.read_csv('/content/train.csv')
test_pd=pd.read_csv('/content/test.csv')
test_label_pd=pd.read_csv('/content/gender_submission.csv')

#drop unuse
train_pd=train_pd.drop(['PassengerId','Name','Cabin','Embarked','Ticket'],axis=1)
test_pd=test_pd.drop(['PassengerId','Name','Cabin','Embarked','Ticket'],axis=1)

#eliminate NaN data
test_pd=pd.concat([test_pd,test_label_pd],axis=1)


train_pd=train_pd.dropna()
test_pd=test_pd.dropna()




train_pd['Age']=(train_pd['Age']-train_pd['Age'].mean())/train_pd['Age'].std()
train_pd['Fare']=(train_pd['Fare']-train_pd['Fare'].mean())/train_pd['Fare'].std()

test_pd['Age']=(test_pd['Age']-test_pd['Age'].mean())/test_pd['Age'].std()
test_pd['Fare']=(test_pd['Fare']-test_pd['Fare'].mean())/test_pd['Fare'].std()

sex_mapping={'female':0,'male':1}
train_pd['Sex']=train_pd['Sex'].map(sex_mapping)
test_pd['Sex']=test_pd['Sex'].map(sex_mapping)

train_pd=train_pd.astype(float)
test_pd=test_pd.astype(float)

train_data=train_pd[['Pclass',  'Age', 'Fare','SibSp','Parch']]
train_label=train_pd['Survived']

test_data=test_pd[['Pclass', 'Age', 'Fare','SibSp','Parch']]
test_label=test_pd['Survived']

train_data=torch.tensor(train_data.values).cuda()
train_label=torch.tensor(train_label.values).reshape(-1,1).cuda()
test_data=torch.tensor(test_data.values).cuda()
test_label=torch.tensor(test_label.values).reshape(-1,1).cuda()
