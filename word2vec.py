import nltk
#lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pandas as pd
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import multiprocessing

df=pd.read_csv(r"C:\Users\KHUSHI CHAUHAN\Documents\project_minor\car_ad_new_1.csv")
df
def clean_data(text):
    text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)
    text = re.sub(r'[\\/×\^\]\[÷]', '', text)
    print("hello\n")
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

def lem(string):
    wnl = WordNetLemmatizer()
    lemmatized_string = ''.join([wnl.lemmatize(words) for words in string])
    return lemmatized_string

''' new_word=" "
 lemmatizer = WordNetLemmatizer()
 # lemmatize string
 for i in string:
     word_tokens = word_tokenize(i)
     # provide context i.e. part-of-speech
     lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
 for j in lemmas:
     new_word=new_word+lemmas
     '''
def get_w2vdf(df):
    w2v_df = pd.DataFrame(df["search"]).values.tolist()
    for i in range(len(w2v_df)):
        w2v_df[i] = w2v_df[i][0].split(" ")
        w2v_df
    return w2v_df

def train_w2v(w2v_df):
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=4,
                         window=4,
                         alpha=0.03,
                         min_alpha=0.0007,
                         sg = 1,
                         workers=cores-1)
   
    w2v_model.build_vocab(w2v_df, progress_per=10000)
    w2v_model.train(w2v_df, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)
    return w2v_model

df[["search"]] = df[["search"]].astype(str)
df["search"] = df["search"].apply(change_lower)
df["search"] = df["search"].apply(remove_numbers)
df["search"] = df["search"].apply(clean_data)
df["search"] = df["search"].apply(remove_punc)
df["search"] = df["search"].apply(remover)
#hello#




print("1")
print(df["search"])
df["search"] = df["search"].apply(lem)
print("2")
print(df["search"])
w2v_df = get_w2vdf(df)
w2v_model = train_w2v(w2v_df)
list=w2v_model.wv.most_similar(positive=["car"])
print(list)
w2v_model
vec=w2v_model.wv['car']
list=w2v_model.wv.index_to_key
print(list)
print(vec)
