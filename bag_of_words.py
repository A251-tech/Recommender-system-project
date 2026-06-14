import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import multiprocessing
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict
#lemmatization
'''
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
'''

df=pd.read_csv(r"C:\Users\ABHINAV CHAUHAN\Documents\project_minor\car_ad_new_1.csv")
df
def clean_data(text):
    text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)
    text = re.sub(r'[\\/×\^\]\[÷]', '', text)
    print(text)
    return text

def remove_numbers(string):
    result = ''.join((element for element in string if not element.isdigit()))
    return result
l=[]
def remove_punc(string):
    punc='''!()-[]{};:'"\, <>./?@#$%^&*_~'''
   
    for ele in string:  
        if ele in punc:  
            string = string.replace(ele, " ")      
    return string

'''def remove_punc(text):
   
    text =  re.sub(r'[^\w\s]', '', text)
    return text'''


def change_lower(text):
    text = text.lower()
    print(text)
    return text

stopwords_list = stopwords.words("english")

def remover(text):
    text_tokens = text.split(" ")
    final_list = [word for word in text_tokens if not word in stopwords_list]
    text = ' '.join(final_list)
    return text
'''
def lem(string):
    wnl = WordNetLemmatizer()
    lemmatized_string = ''.join([wnl.lemmatize(words) for words in string])
    return lemmatized_string
'''
df[["search"]] = df[["search"]].astype(str)
df["search"] = df["search"].apply(change_lower)
df["search"] = df["search"].apply(remove_numbers)
df["search"] = df["search"].apply(clean_data)
df["search"] = df["search"].apply(remove_punc)
df["search"] = df["search"].apply(remover)
#df["search"] = df["search"].apply(lem)

sentences = []
vocab = []
list=df['search']
print(list)
for text in list :
    x = word_tokenize(text)
    sentence = [w.lower() for w in x if w.isalpha() ]
    sentences.append(sentence)
    for word in sentence:
        if word not in vocab:
            vocab.append(word)
 
len_vector = len(vocab)
index_word = {}
i = 0
for word in vocab:
    index_word[word] = i
    i += 1
def bag_of_words(text):
    count_dict = defaultdict(int)
    vec = np.zeros(len_vector)
    for item in text:
        count_dict[item] += 1
    for key,item in count_dict.items():
        vec[index_word[key]] = item
    return vec  
vector_1=[]
for i in sentences:
 vector = bag_of_words(i)
 vector_1.append(vector)
 print(vector)


'''# cosine formula
for i in range(len(rvector)):
        c+= l1[i]*l2[i]
cosine = c / float((sum(l1)*sum(l2))**0.5)
print("similarity: ", cosine)


'''

import numpy as np
'''def cosine_similarity(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim
pairwise_similarity = np.identity(len(vector_1))

for i in range(len(vector_1)):
    for j in range(i+1, len(vector_1)):
        i=i[:,0].astype(int)
        j=j[:,0].astype(int)
        pairwise_similarity[i][j] = cosine_similarity(vector_1[i], vector_1[j])
        pairwise_similarity[j][i] = pairwise_similarity[i][j]
for i in pairwise_similarity:
    for j in pairwise_similarity:
        print(pairwise_similarity[i][j])
'''
summ=0.0
s1=[]
for i in range(len(vector_1)):
    for j  in range(i+1,len(vector_1)):
        c=0
        v1=vector_1[i]
        v2=vector_1[j]
        #for k in range(len(vector)):
        c+= vector_1[i]*vector_1[j]
        cosine = c / float((sum(v1)*sum(v2))**0.5)
        summ=summ+cosine
summ=summ/len(list)
print("cosine similarity: ", summ)
s1.append(summ)
#print(vector_1)
#shape = vector_1.shape
y_pred=[]
no_of_col=len(vector)
for i in range(no_of_col):
    y_pred.append(1)
print(s1)
print(y_pred)
'''
def compute_accuracy(y_true, y_pred):
    correct_predictions = 0
    # iterate over each label and check
    for true, predicted in zip(y_true, y_pred):
        if true.any() == predicted.any():
            correct_predictions += 1
    # compute the accuracy
    accuracy = correct_predictions/len(y_true)
    return accuracy
print(compute_accuracy(s1,y_pred))'''
'''
from sklearn.metrics import accuracy_score
s1[:,None]
y_pred[None, :]
print(accuracy_score(s1, y_pred))'''
 
 
