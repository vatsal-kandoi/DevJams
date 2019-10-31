#!/usr/bin/env python
# coding: utf-8

# In[14]:


# import csv                          
# from sklearn.svm import LinearSVC
# from nltk.classify import SklearnClassifier
# from random import shuffle
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import accuracy_score
# import numpy as np
# import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')


# In[15]:


# load data from a file and append it to the rawData
# def loadData(path, Text=None):
#     with open(path) as f:
#         reader = csv.reader(f, delimiter=',')
#         next(reader)
#         for line in reader:
#             (Id, Text, Label) = parseReview(line)
#             rawData.append((Id, Text, Label))
#             preprocessedData.append((Id, preProcess(Text), Label))
        
# def splitData(percentage):
#     dataSamples = len(rawData)
#     halfOfData = int(len(rawData)/2)
#     trainingSamples = int((percentage*dataSamples)/2)
#     for (_, Text, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
#         trainData.append((toFeatureVector(preProcess(Text)),Label))
#     for (_, Text, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
#         testData.append((toFeatureVector(preProcess(Text)),Label))


# # In[16]:


def parseReview(reviewLine):
    s=""
    if reviewLine[1]=="__label1__":
        s = "fake"
    else: 
        s = "real"
    return (reviewLine[0], reviewLine[8], s)


# In[17]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION

def preProcess(text):
    return word_tokenize(text)


# In[18]:


featureDict = {} # A global dictionary of features

def toFeatureVector(tokens):
    localDict = {}
    for token in tokens:
        if token not in featureDict:
            featureDict[token] = 1
        else:
            featureDict[token] = +1
   
        if token not in localDict:
            localDict[token] = 1
        else:
            localDict[token] = +1
    
    return localDict


# In[19]:


# # TRAINING AND VALIDATING OUR CLASSIFIER
# def trainClassifier(trainData):
#     print("Training Classifier...")
#     pipeline =  Pipeline([('svc', LinearSVC())])
#     return SklearnClassifier(pipeline).train(trainData)


# In[71]:


# def crossValidate(dataset, folds):
#     shuffle(dataset)
#     cv_results = []
#     foldSize = int(len(dataset)/folds)
#     for i in range(0,len(dataset),foldSize):
#         classifier = trainClassifier(dataset[:i]+dataset[foldSize+i:])
#         y_pred = predictLabels(dataset[i:i+foldSize],classifier)
#         a = accuracy_score(list(map(lambda d : d[1], dataset[i:i+foldSize])), y_pred)
#         (p,r,f,_) = precision_recall_fscore_support(list(map(lambda d : d[1], dataset[i:i+foldSize])), y_pred, average ='macro')
#         #print(a,p,r,f)
#         cv_results.append((a,p,r,f))
#     cv_results = (np.mean(np.array(cv_results),axis=0))
#     return cv_results


# In[21]:


# PREDICTING LABELS GIVEN A CLASSIFIER

def predictLabels(reviewSamples, classifier):
    print(reviewSamples)
    return classifier.classify_many(map(lambda t: t, reviewSamples))

def predictLabel(reviewSample, classifier):
    return classifier.classify(toFeatureVector(preProcess(reviewSample)))


# In[23]:


# MAIN

# loading reviews
# rawData = []          # the filtered data from the dataset file (should be 21000 samples)
# preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
# trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
# testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# # the output classes
# fakeLabel = 'fake'
# realLabel = 'real'

# # references to the data files
# reviewPath = 'amazon_reviews.txt'

# ## Do the actual stuff
# # We parse the dataset and put it in a raw data list
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Preparing the dataset...",sep='\n')
# loadData(reviewPath) 
# # We split the raw dataset into a set of training data and a set of test data (80/20)
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Preparing training and test data...",sep='\n')
# splitData(0.8)
# # We print the number of training samples and the number of features
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')
# print("Mean of cross-validations (Accuracy, Precision, Recall, Fscore): ", crossValidate(trainData, 10))


# ====== INCLUDING LEMMATIZATION,REMOVING STOP WORDS AND PUNCTUATIONS ======

# In[32]:


# from nltk.corpus import stopwords
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem import WordNetLemmatizer
# from nltk.util import ngrams
# import string


# In[33]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION

# table = str.maketrans({key: None for key in string.punctuation})

def preProcess(text):
    # Should return a list of tokens
    lemmatizer = WordNetLemmatizer()
    filtered_tokens=[]
    stop_words = set(stopwords.words('english'))
    text = text.translate(table)
    for w in text.split(" "):
        if w not in stop_words:
            filtered_tokens.append(lemmatizer.lemmatize(w.lower()))
    return filtered_tokens


# In[35]:


# MAIN

# loading reviews
# rawData = []          # the filtered data from the dataset file (should be 21000 samples)
# preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
# trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
# testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# # the output classes
# fakeLabel = 'fake'
# realLabel = 'real'

# # references to the data files
# reviewPath = 'amazon_reviews.txt'

# ## Do the actual stuff
# # We parse the dataset and put it in a raw data list
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Preparing the dataset...",sep='\n')
# loadData(reviewPath) 
# # We split the raw dataset into a set of training data and a set of test data (80/20)
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Preparing training and test data...",sep='\n')
# splitData(0.8)
# # We print the number of training samples and the number of features
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')
# print("Mean of cross-validations (Accuracy, Precision, Recall, Fscore): ", crossValidate(trainData, 10))


# ====== INTRODUCING THE BIGRAMS & TRYING DIFFERENT VALUES OF C IN LINEARSVC FUNCTION ======

# In[45]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION
# Input: a string of one review
# table = str.maketrans({key: None for key in string.punctuation})
def preProcess(text):
    # Should return a list of tokens
    lemmatizer = WordNetLemmatizer()
    filtered_tokens=[]
    lemmatized_tokens = []
    stop_words = set(stopwords.words('english'))
    text = text.translate(table)
    for w in text.split(" "):
        if w not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))
        filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens
    return filtered_tokens


# In[59]:


# TRAINING AND VALIDATING OUR CLASSIFIER
def trainClassifier(trainData):
    print("Training Classifier...")
    pipeline =  Pipeline([('svc', LinearSVC(C=0.01))])
    return SklearnClassifier(pipeline).train(trainData)


# In[60]:


# MAIN

# loading reviews
# rawData = []          # the filtered data from the dataset file (should be 21000 samples)
# preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
# trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
# testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# # the output classes
# fakeLabel = 'fake'
# realLabel = 'real'

# # references to the data files
# reviewPath = 'amazon_reviews.txt'

# ## Do the actual stuff
# # We parse the dataset and put it in a raw data list
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Preparing the dataset...",sep='\n')
# loadData(reviewPath) 
# # We split the raw dataset into a set of training data and a set of test data (80/20)
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Preparing training and test data...",sep='\n')
# splitData(0.8)
# # We print the number of training samples and the number of features
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')
# print("Mean of cross-validations (Accuracy, Precision, Recall, Fscore): ", crossValidate(trainData, 10))


# ====== TAKING EXTRA FEATURES (RATING, VERIFIED PURCHASE, PRODUCT CATEGORY) ======

# In[66]:


# load data from a file and append it to the rawData
def loadData(path, Text=None):
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for line in reader:
            (Id, Rating, verified_Purchase, product_Category, Text, Label) = parseReview(line)
            rawData.append((Id, Rating, verified_Purchase, product_Category, Text, Label))
            #preprocessedData.append((Id, preProcess(Text), Label))
        
def splitData(percentage):
    dataSamples = len(rawData)
    halfOfData = int(len(rawData)/2)
    trainingSamples = int((percentage*dataSamples)/2)
    for (_, Rating, verified_Purchase, product_Category, Text, Label) in rawData[:trainingSamples] + rawData[halfOfData:halfOfData+trainingSamples]:
        trainData.append((toFeatureVector(Rating, verified_Purchase, product_Category, preProcess(Text)),Label))
    for (_, Rating, verified_Purchase, product_Category, Text, Label) in rawData[trainingSamples:halfOfData] + rawData[halfOfData+trainingSamples:]:
        testData.append((toFeatureVector(Rating, verified_Purchase, product_Category, preProcess(Text)),Label))


# In[67]:


# QUESTION 1

# Convert line from input file into an id/text/label tuple
def parseReview(reviewLine):
    # Should return a triple of an integer, a string containing the review, and a string indicating the label
    s=""
    if reviewLine[1]=="__label1__":
        s = "fake"
    else: 
        s = "real"
    return (reviewLine[0], reviewLine[2], reviewLine[3],reviewLine[4], reviewLine[8], s)


# In[68]:


# TEXT PREPROCESSING AND FEATURE VECTORIZATION
# Input: a string of one review
# table = str.maketrans({key: None for key in string.punctuation})
# def preProcess(text):
#     # Should return a list of tokens
#     lemmatizer = WordNetLemmatizer()
#     filtered_tokens=[]
#     lemmatized_tokens = []
#     stop_words = set(stopwords.words('english'))
#     text = text.translate(table)
#     for w in text.split(" "):
#         if w not in stop_words:
#             lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))
#         filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens
#     return filtered_tokens


# In[69]:


# featureDict = {} # A global dictionary of features

# def toFeatureVector(Rating, verified_Purchase, product_Category, tokens):
#     localDict = {}
    
# #Rating

#     #print(Rating)
#     featureDict["R"] = 1   
#     localDict["R"] = Rating

# #Verified_Purchase
  
#     featureDict["VP"] = 1
            
#     if verified_Purchase == "N":
#         localDict["VP"] = 0
#     else:
#         localDict["VP"] = 1

# #Product_Category

    
#     if product_Category not in featureDict:
#         featureDict[product_Category] = 1
#     else:
#         featureDict[product_Category] = +1
            
#     if product_Category not in localDict:
#         localDict[product_Category] = 1
#     else:
#         localDict[product_Category] = +1
            
            
# #Text        

#     for token in tokens:
#         if token not in featureDict:
#             featureDict[token] = 1
#         else:
#             featureDict[token] = +1
            
#         if token not in localDict:
#             localDict[token] = 1
#         else:
#             localDict[token] = +1
    
#     return localDict


# In[70]:


# MAIN

# loading reviews
# rawData = []          # the filtered data from the dataset file (should be 21000 samples)
# preprocessedData = [] # the preprocessed reviews (just to see how your preprocessing is doing)
# trainData = []        # the training data as a percentage of the total dataset (currently 80%, or 16800 samples)
# testData = []         # the test data as a percentage of the total dataset (currently 20%, or 4200 samples)

# # the output classes
# fakeLabel = 'fake'
# realLabel = 'real'

# # references to the data files
# reviewPath = 'amazon_reviews.txt'

# ## Do the actual stuff
# # We parse the dataset and put it in a raw data list
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Preparing the dataset...",sep='\n')
# loadData(reviewPath) 
# # We split the raw dataset into a set of training data and a set of test data (80/20)
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Preparing training and test data...",sep='\n')
# splitData(0.8)
# # We print the number of training samples and the number of features
# print("Now %d rawData, %d trainData, %d testData" % (len(rawData), len(trainData), len(testData)),
#       "Training Samples: ", len(trainData), "Features: ", len(featureDict), sep='\n')
# print("Mean of cross-validations (Accuracy, Precision, Recall, Fscore): ", crossValidate(trainData, 10))


# In[72]:


#  TEST DATA
# classifier = trainClassifier(trainData)
# predictions = predictLabels(testData, classifier)
# true_labels = list(map(lambda d: d[1], testData))
# a = accuracy_score(true_labels, predictions)
# p, r, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
# print("accuracy: ", a)
# print("Precision: ", p)
# print("Recall: ", a)
# print("f1-score: ", f1)
# classifier = trainClassifier(trainData)
import pickle 
  
# Save the trained model as a pickle string. 
# saved_model = pickle.dumps(classifier) 
from sklearn.externals import joblib 
  
# # Save the model as a pickle in a file 
# joblib.dump(classifier, 'filename.pkl') 

classifier = joblib.load('filename.pkl')    

def pred(obj):
    predictions = predictLabels(obj['reviews'], classifier)
    # true_labels = list(map(lambda d: d[''], obj['reviews']))
    # a = accuracy_score(true_labels, predictions,normalize=False)
    # obj.accuracy = a;
    obj['real']=[];
    obj['fake']=[];
    print(predictions)
    for i in range(len(predictions)):
        if predictions[i]=='real':
            obj['real'].append(obj['reviews'][i]);
        else:
            obj['fake'].append(obj['reviews'][i]);
    newrating=0;
    for i in range(len(obj['real'])):
        newrating+=float(obj['real'][i]['review_rating'])
    if len(obj['real'])>0:
        newrating=newrating/len(obj['real'])
    obj['newrating']=newrating;
    return obj;
    
