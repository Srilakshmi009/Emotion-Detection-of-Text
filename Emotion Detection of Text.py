#!/usr/bin/env python
# coding: utf-8

# # Text to EMotion

# In[34]:


import pandas as pd
import numpy as np


# In[35]:


import matplotlib.pyplot as plb
import seaborn as sns


# In[59]:


df = pd.read_csv("Desktop/Data1.csv")


# In[60]:


df.head()


# In[61]:


#shape
df.shape


# In[62]:


#data types
df.dtypes


# In[63]:


#check for missing values
df.isnull().sum()


# In[64]:


#value count of emotions
df['sentiment'].value_counts()


# In[65]:


#value of count of emotion
df['sentiment'].value_counts().plot(kind='bar')


# In[66]:


#seaborn to plot
plb.figure(figsize=(15,10))
sns.countplot(x='sentiment',data=df)
plb.show()


# # Exploration
# + Sentiment Analysis
# + Keyword Extraction
#     - keywords for each emotion
#     - wordcloud

# In[67]:


pip install textblob


# In[68]:


#sentiment analysis
from textblob import TextBlob


# In[69]:


def get_sentiment(text):
    blob = TextBlob(text)
    TypeofEmotion = blob.sentiment.polarity
    if TypeofEmotion > 0:
        result = "Positive"
    elif TypeofEmotion < 0:
        result = "Negative"
    else:
        result = "Neutral"
    return result


# In[70]:


#test fxn
get_sentiment("I love coding")


# In[72]:


df['TypeofEmotion']=df['Clean_Text'].apply(get_sentiment)


# In[73]:


df.head()


# In[ ]:


#compare our Sentiment vs Type of emotion
df.groupby(['sentiment','TypeofEmotion']).size()


# In[ ]:


#first method:using matplotlib
#compare our Sentiment vs Type of emotion
df.groupby(['sentiment','TypeofEmotion']).size().plot(kind='bar')


# In[ ]:


#using seaborn
sns.factorplot
sns.catplot


# In[ ]:


sns.catplot(x='sentiment',hue='TypeofEmotion',data=df,kind='count',aspect=1.5)


# # Keyword Extraction
# + Extract most commonest words per class of emotion

# In[74]:


from collections import Counter


# In[75]:


def extract_keywords(Text,num=50):
    tokens = [ tok for tok in Text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)


# In[76]:


# List of Emotion
emotion_list = df['sentiment'].unique().tolist()


# In[77]:


emotion_list


# In[78]:


fun_list = df[df['sentiment'] == 'fun']['Clean_Text'].tolist()


# In[79]:


#fun document
fun_docx = ' '.join(fun_list)


# In[80]:


fun_docx


# In[81]:


#extract keywords
keyword_fun = extract_keywords(fun_docx)


# In[83]:


keyword_fun


# In[93]:


#plot
def plot_most_common_words(mydict,emotion_name):
    df_01 = pd.DataFrame(mydict.items(),columns=['token','count'])
    plb.figure(figsize=(20,10))
    plb.title("plot of {} Most common keywords".format(emotion_name))
    sns.barplot(x='token',y='count',data=df_01)
    plb.xticks(rotation=45)
    plb.show()


# In[94]:


plot_most_common_words(keyword_fun,"fun")


# In[95]:


surprise_list = df[df['sentiment'] == 'surprise']['Clean_Text'].tolist()
surprise_docx = ' '.join(surprise_list)
keyword_surprise = extract_keywords(surprise_docx)


# In[96]:


plot_most_common_words(keyword_surprise,"surprise")


# In[106]:


import sys
print(sys.executable)


# In[126]:


from wordcloud import WordCloud


# In[128]:


def plot_wordcloud(docx):
    mywordcloud = WordCloud().generate(docx)
    plb.figure(figsize=(20,10))
    plb.imshow(mywordcloud,interpolation='bilinear')
    plb.axis('off')
    plb.show()


# In[130]:


plot_wordcloud(fun_docx)


# In[131]:


plot_wordcloud(surprise_docx)


# # machine learning
# + Naive Bayes
# + Logistic Regression
# + KNN
# + Decision Tree
# 
# # Compare with sparkNLP/NLU john snows lab

# In[137]:


# load ML packages
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

#Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix


# In[138]:


#split dataset
from sklearn.model_selection import train_test_split


# In[139]:


#build features from text


# In[140]:


Xfeatures = df['Clean_Text']
ylabels = df['sentiment']


# In[141]:


Xfeatures


# In[143]:


#vectorizer
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)


# In[144]:


#Get features by name
cv.get_feature_names()


# In[146]:


#to dense array (numpy)
X.toarray()


# In[147]:


#split dataset
X_train,X_test,y_train,y_test = train_test_split(X,ylabels,test_size=0.3,random_state=42)


# # build model
# 

# In[148]:


nv_model = MultinomialNB()
nv_model.fit(X_train,y_train)


# In[150]:


#Accuracy
#method 1
nv_model.score(X_test,y_test)


# In[151]:


#predications
y_pred_for_nv = nv_model.predict(X_test)


# In[152]:


y_pred_for_nv


# # Make A single prediction
# + Vectorized our Text
# + Applied our Model

# In[153]:


sample_text = ["I love coding so much"]


# In[154]:


vect = cv.transform(sample_text).toarray()


# In[155]:


#make prediction
nv_model.predict(vect)


# In[156]:


# Check for the prediction probability(percentage)/confidence score
nv_model.predict_proba(vect)


# In[157]:


#get all class for model
nv_model.classes_


# In[158]:


np.max(nv_model.predict_proba(vect))


# In[164]:


def predict_sentiment(sample_text,model):
    myvect = cv.transform(sample_text).toarray()
    prediction = model.predict(myvect)
    pred_proba = model.predict_proba(myvect)
    pred_percentage_for_all = dict(zip(model.classes_,pred_proba[0]))
    print("Prediction:{},Prediction Score:{}".format(prediction[0],np.max(pred_proba)))
    print(prediction[0])
    return pred_percentage_for_all


# In[165]:


predict_sentiment(sample_text,nv_model)


# In[167]:


predict_sentiment(["She hates crying all day"],nv_model)


# # Model Evaluation

# In[170]:


# Classification
print(classification_report(y_test,y_pred_for_nv))


# In[172]:


# confusion 
confusion_matrix(y_test,y_pred_for_nv)


# In[173]:


#plot confusion matrix
plot_confusion_matrix(nv_model,X_test,y_test)


# #  Model Interpretation
#     . Eli5
#     . Lime
#     . Shap

# In[175]:


# logistic regression
lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)


# In[176]:


#Accuracy
lr_model.score(X_test,y_test)


# In[178]:


#Single Predict
predict_sentiment(sample_text,lr_model)


# In[181]:


#Interpret Model
import eli5


# In[182]:


# show the weights for each class/label
eli5.show_weights(lr_model,top=20)


# In[183]:


class_names = ylabels.unique().tolist()


# In[193]:


feature_names = cv.get_feature_names()


# In[194]:


eli5.show_weights(lr_model,feature_names=feature_names,target_names=class_names)


# In[189]:




