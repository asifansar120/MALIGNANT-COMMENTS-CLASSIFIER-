#!/usr/bin/env python
# coding: utf-8

# # MALIGNANT COMMENTS CLASSIFIER

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#load datasets
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# In[3]:


train


# In[4]:


test


# In[5]:


print('Train_data:',train.shape)
print('Test_data:',test.shape)


# In[6]:


train.columns


# In[7]:


test.columns


# In[8]:


train.isnull().sum()


# In[9]:


test.isnull().sum()


# # Visualizations

# Count plot:

# In[14]:


col=['malignant','highly_malignant','rude','threat','abuse','loathe']

for i in col:
    print(i)
    print("\n")
    print(train[i].value_counts())
    sns.countplot(train[i])
    plt.show()


# From the above observation we can see that, 15294 comments are malignant and 144277 comments are not malignant.
# 1595 comments are highly_malignant and 157976 comments are not highly_malignant.
# 8449 comments are rude and 151122 are not rude.
# 478 comments are threat and 159093 comments are not threat.
# 7877 comments are abuse and 151694 comments are not abuse.
# 1405 comments are loathe and 158166 comments are not loathe.

# # Correlation:

# In[15]:


train.corr()


# correlation using heatmap:

# In[16]:


plt.figure(figsize=(15,7))
sns.heatmap(train.corr(),annot=True,linewidth=0.5,linecolor="black",fmt='.2f')


# # Describing dataset:

# In[17]:


train.describe()


# Describe of dataset using heatmap:

# In[18]:


plt.figure(figsize=(15,12))
sns.heatmap(round(train.describe()[1:].transpose(),2),linewidth=2,annot=True,fmt="f")
plt.xticks(fontsize=18)
plt.yticks(fontsize=12)
plt.title("Variable Summary")
plt.show()

From the above plotting, we can determine mean, standard deviation, minimum and maximum value of each column of the dataset.

159571 rows
8 columns

malignant:
    mean=0.095844
    std=0.294379
    max_value=1.000000
    min_value=0.000000
    
highly_malignant:
    mean=0.009996
    std=0.099477
    max_value=1.000000
    min_value=0.000000
    
rude:
    mean=0.052948
    std=0.223931
    max_value=1.000000
    min_value=0.000000
    
threat:
    mean=0.002996
    std=0.054650
    max_value=1.000000
    min_value=0.000000
    
abuse:
    mean=0.049364
    std=0.216627
    max_value=1.000000
    min_value=0.000000
    
loathe:
    mean=0.008805
    std=0.093420
    max_value=1.000000
    min_value=0.000000
# In[21]:


print(train.info())
print("\n")
print(test.info())


# # Skewness:

# In[36]:


train.skew()


# All the columns of the dataset has skewness.

# # Normal Distribution Curve:

# In[37]:


sns.distplot(train['malignant'])


# The data of the column is not normalized. The building blocks is out of the normalized curve.

# In[38]:


sns.distplot(train['highly_malignant'])


# The data of the column is not normalized. The building blocks is out of the normalized curve.

# In[39]:


sns.distplot(train['rude'])


# The data of the column is not normalized. The building blocks is out of the normalized curve.

# In[40]:


sns.distplot(train['threat'])


# The data of the column is not normalized. The building blocks is out of the normalized curve.

# In[41]:


sns.distplot(train['abuse'])


# The data of the column is not normalized. The building blocks is out of the normalized curve.

# In[42]:


sns.distplot(train['loathe'])


# The data of the column is not normalized. The building blocks is out of the normalized curve.

# In[5]:


#importing necessary libraries

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string


# In[6]:


train['length']=train['comment_text'].str.len()
train


# In[7]:


#convert all messages to lower case
train['comment_text'] = train['comment_text'].str.lower()

#Replace email address with 'email'
train['comment_text'] = train['comment_text'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')

#Replace URLs with 'webaddress'
train['comment_text'] = train['comment_text'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

#Replace money sumbols with 'moneysymb'
train['comment_text'] = train['comment_text'].str.replace(r'£|\$', 'dollars')

#Replace 10 digit phone numbers with 'phonenumber'
train['comment_text'] = train['comment_text'].str.replace(r'\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumber')

#Replace numbers with 'num'
train['comment_text'] = train['comment_text'].str.replace(r'\d+(\.\d+)?', 'num')

train['comment_text'] = train['comment_text'].apply(lambda x: ''.join(term for term in x.split() if term not in string.punctuation))

stop_words = set(stopwords.words('english') + ['u','ur','4','2','im','dont','doin','ure'])
train['comment_text'] = train['comment_text'].apply(lambda x: ''.join(term for term in x.split() if term not in stop_words))

lem=WordNetLemmatizer()
train['comment_text'] = train['comment_text'].apply(lambda x: ''.join(lem.lemmatize(t) for t in x.split()))


# In[8]:


train['clean_length']=train.comment_text.str.len()
train


# In[9]:


#total length removal
print('Original Length: ', train.length.sum())
print('Clean Length: ', train.clean_length.sum())


# In[10]:


#!pip install wordcloud


# In[11]:


#offensive loud words
from wordcloud import WordCloud
hams = train['comment_text'][train['malignant']==1]
spam_cloud = WordCloud(width=600, height=400, background_color='white',max_words=50).generate(''.join(hams))
plt.figure(figsize=(10,8),facecolor='r')
plt.imshow(spam_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[12]:


target_column=['malignant','highly_malignant','rude','threat','abuse','loathe']

target=train[target_column]

train['bad']=train[target_column].sum(axis=1)
print(train['bad'].value_counts())
train['bad']=train['bad']>0
train['bad']=train['bad'].astype(int)
print(train['bad'].value_counts())


# In[13]:


sns.countplot(x='bad', data=train)
print(train['bad'].value_counts())
plt.show()


# In[14]:


train.head()


# In[15]:


#converting text into vectors using TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer(max_features=10000, stop_words='english')
feature= tfidf.fit_transform(train['comment_text'])


# In[16]:


print(train.shape)
print(test.shape)


# # Dividing Dataframe into feature and target

# In[17]:


x=feature
y=train['bad']


# In[18]:


y.value_counts()


# In[19]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
xtrain,ytrain=sm.fit_resample(x,y)


# In[20]:


ytrain.value_counts()


# # Model Building

# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# # Logistic Regression:

# In[29]:


lg=LogisticRegression()

train_x,test_x,train_y,test_y=train_test_split(xtrain,ytrain,test_size=0.20,random_state=45)
lg.fit(train_x,train_y)
pred_train=lg.predict(train_x)
pred_test=lg.predict(test_x)

print("Training Accuracy:",accuracy_score(train_y,pred_train)*100)
print("Testing Accuracy:",accuracy_score(test_y,pred_test)*100)


# Cross Validation for logistic regression:

# In[30]:


from sklearn.model_selection import cross_val_score

lg.fit(train_x,train_y)
lg.score(train_x,train_y)
pred_lg = lg.predict(test_x)

lss = accuracy_score(test_y,pred_lg)
for j in range(2,10):
    lsscore = cross_val_score(lg,xtrain,ytrain,cv=j)
    ls_cv = lsscore.mean() 
    print("At cv:-",j)
    print("Cross validation score is:-",ls_cv*100 )
    print("Accuracy score is :-",lss*100)
    print("\n")


# In[31]:


print("At cv:",9)
print("Cross validation score is:",74.33099030511153)
print("Accuracy score is :",74.33858281448926)


# AUC-ROC Curve:

# In[32]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds=roc_curve(pred_lg,test_y)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=10,label='ROC Curve(area=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=10,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LOGISTIC REGRESSION')
plt.legend(loc="lower right")
plt.show()


# # Approaching more classifiers:

# In[33]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# # 1.Decision Tree Classifier:

# In[34]:


parameters = {'criterion':['gini', 'entropy'],'splitter':['best','random']}
dtc = DecisionTreeClassifier()
clf = GridSearchCV(dtc,parameters)
clf.fit(train_x,train_y)

print(clf.best_params_)


# In[36]:


dtc = DecisionTreeClassifier(criterion='gini', splitter='best')
dtc.fit(train_x,train_y)
dtc.score(train_x,train_y)
pred_dtc = dtc.predict(test_x)

print("Accuracy Score:",accuracy_score(test_y,pred_dtc)*100)
print("Classification report:",classification_report(test_y,pred_dtc)*100)
print("Confusion Matrix:",confusion_matrix(test_y,pred_dtc)*100)

dtc_score = cross_val_score(dtc,xtrain,ytrain,cv=9)
dtc_cc = dtc_score.mean() 
print('Cross Val Score:',dtc_cc*100)


# AUC-ROC Curve:

# In[37]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds=roc_curve(pred_dtc,test_y)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=10,label='ROC Curve(area=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=10,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree Classifier')
plt.legend(loc="lower right")
plt.show()


# # 2.KNeighbors Classifier:

# In[38]:


parameters = {'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
knc = KNeighborsClassifier()
clf = GridSearchCV(knc,parameters)
clf.fit(train_x,train_y)

print(clf.best_params_)


# In[39]:


knc = KNeighborsClassifier(algorithm='auto', n_neighbors=5)
knc.fit(train_x,train_y)
knc.score(train_x,train_y)
pred_knc = knc.predict(test_x)

print("Accuracy Score:",accuracy_score(test_y,pred_knc)*100)
print("Classification report:",classification_report(test_y,pred_knc)*100)
print("Confusion Matrix:",confusion_matrix(test_y,pred_knc)*100)

knc_score = cross_val_score(knc,xtrain,ytrain,cv=9)
knc_cc = knc_score.mean() 
print('Cross Val Score:',knc_cc*100)


# AUC-ROC Curve:

# In[40]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds=roc_curve(pred_knc,test_y)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=10,label='ROC Curve(area=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=10,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNeighbors Classifier')
plt.legend(loc="lower right")
plt.show()


# # ENSEMBLE METHODS:

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# # 1.Random Forest Classifier:

# In[42]:


parameters = {'criterion':['gini', 'entropy'],'n_estimators':[100]}
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc,parameters)
clf.fit(train_x,train_y)

print(clf.best_params_)


# In[43]:


rfc = RandomForestClassifier(criterion='entropy', n_estimators=100)
rfc.fit(train_x,train_y)
rfc.score(train_x,train_y)
pred_rfc = rfc.predict(test_x)

print("Accuracy Score:",accuracy_score(test_y,pred_rfc)*100)
print("Classification report:",classification_report(test_y,pred_rfc)*100)
print("Confusion Matrix:",confusion_matrix(test_y,pred_rfc)*100)

rfc_score = cross_val_score(rfc,xtrain,ytrain,cv=9)
rfc_cc = rfc_score.mean() 
print('Cross Val Score:',rfc_cc*100)


# AUC-ROC Curve

# In[44]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds=roc_curve(pred_rfc,test_y)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=10,label='ROC Curve(area=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=10,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()


# # 2.Ada Boost Classifier:

# In[45]:


abc = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME')
abc.fit(train_x,train_y)
abc.score(train_x,train_y)
pred_abc = abc.predict(test_x)

print("Accuracy Score:",accuracy_score(test_y,pred_abc)*100)
print("Classification report:",classification_report(test_y,pred_abc)*100)
print("Confusion Matrix:",confusion_matrix(test_y,pred_abc)*100)

abc_score = cross_val_score(abc,xtrain,ytrain,cv=9)
abc_cc = abc_score.mean() 
print('Cross Val Score:',abc_cc*100)


# AUC-ROC Curve

# In[46]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds=roc_curve(pred_abc,test_y)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=10,label='ROC Curve(area=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=10,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ada Boost Classifier')
plt.legend(loc="lower right")
plt.show()


# # 3.Gradient Boosting Classifier:

# In[47]:


gbc = GradientBoostingClassifier(criterion='mse', n_estimators=100, learning_rate=0.1, loss='deviance')
gbc.fit(train_x,train_y)
gbc.score(train_x,train_y)
pred_gbc = gbc.predict(test_x)

print("Accuracy Score:",accuracy_score(test_y,pred_gbc)*100)
print("Classification report:",classification_report(test_y,pred_gbc)*100)
print("Confusion Matrix:",confusion_matrix(test_y,pred_gbc)*100)

gbc_score = cross_val_score(gbc,xtrain,ytrain,cv=9)
gbc_cc = gbc_score.mean() 
print('Cross Val Score:',gbc_cc*100)


# AUC-ROC Curve:

# In[48]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds=roc_curve(pred_gbc,test_y)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=10,label='ROC Curve(area=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=10,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Gradient Boosting Classifier')
plt.legend(loc="lower right")
plt.show()


# # XGBOOST:

# In[50]:


#!pip install xgboost


# In[51]:


import xgboost

xgb = xgboost.XGBClassifier()
xgb.fit(train_x,train_y)
xgb.score(train_x,train_y)
pred_xgb = xgb.predict(test_x)

print("Accuracy Score:",accuracy_score(test_y,pred_xgb)*100)
print("Classification report:",classification_report(test_y,pred_xgb)*100)
print("Confusion Matrix:",confusion_matrix(test_y,pred_xgb)*100)

xgb_score = cross_val_score(xgb,xtrain,ytrain,cv=9)
xgb_cc = gbc_score.mean() 
print('Cross Val Score:',xgb_cc*100)


# AUC-ROC Curve:

# In[52]:


from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds=roc_curve(pred_xgb,test_y)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=10,label='ROC Curve(area=%0.2f)'%roc_auc)
plt.plot([0,1],[0,1],color='navy',lw=10,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Classifier')
plt.legend(loc="lower right")
plt.show()


# # The best model is Random Forest Classifier. Since the difference between the percentage score of cross validation,accuracy_score and ROC Curve is optimum.

# # Model Saving:

# In[55]:


import pickle
filename='malign-comment.pkl'
pickle.dump(rfc,open(filename,'wb'))


# # Coclusion:

# In[56]:


import numpy as np
a=np.array(test_y)
predicted=np.array(rfc.predict(test_x))
mal_con=pd.DataFrame({"original":a,"predicted":predicted},index=range(len(a)))
mal_con


# From the above table, the model is predicted with 78 percent accuracy.
# 

# In[57]:


#Converting test dataset into vector

test1=tfidf.fit_transform(test['comment_text'])
test1


# In[58]:


#predicting test dataset using chosen model

pred = rfc.predict(test1)
pred


# In[59]:


#creating dataset with predicted output

test['Predicted Output'] = pred
test


# In[65]:


#converting into csv file

test.to_csv('Test.csv', index = False)


# In[67]:


#downloading the csv file

test.to_csv(r'C:\Users\DELL\Downloads\Test.csv')

