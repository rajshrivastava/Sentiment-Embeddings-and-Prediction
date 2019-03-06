#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:58:50 2019
SENSITIVITY CLASSIFICATION - Testing
@author: Raj Kumar Shrivastava
"""
import pandas as pd
import numpy as np
import pickle
import time
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import tensorflow as tf

class NeuralNet():
    def __init__(self):                           #loading trained parameters
        
        self.w1   = np.load('w1.npy') 
        self.w2   = np.load('w2.npy')       #lookup->linear1
        self.b2   = np.load('b2.npy')
        self.w3   = np.load('w3.npy')       #linear1->htanh
        self.b3   = np.load('b3.npy')
        self.w4_s = np.load('w4_s.npy')     #htanh->linear2_s
        self.b4_s = np.load('b4_s.npy')
        self.w5   = np.load('w5.npy')       #linear2_s->softmax
        self.b5   = np.load('b5.npy')
        
        self.n = len(self.w1[0])            #embedding size
        self.voc_size = len(self.w1)
        self.window = int((len(self.w2)/self.n - 1)/2)
        self.m = len(self.b2)
        self.word_index = pickle.load(open('word_index.txt','rb'))
    
    def create_corpus(self, mails, polarity):            #test data cleansing
        mails=list(mails)
        stop_words=['k','m','t','d','e','f','g','h','i','u','r','I','im',\
                    'ourselves', 'hers', 'between', 'yourself', 'again', \
                    'there', 'about', 'once', 'during', 'out', 'very', \
                    'having', 'with', 'they', 'own', 'an', 'be', 'some', \
                    'for', 'do', 'its', 'yours', 'such', 'into', 'of', \
                    'most', 'itself', 'other', 'off', 'is', 's', 'am', \
                    'or', 'who', 'as', 'from', 'him', 'each', 'the', \
                    'themselves', 'until', 'below', 'are', 'we', 'these', \
                    'your', 'his', 'through', 'don', 'nor', 'me', 'were', \
                    'her', 'more', 'himself', 'this', 'should', 'our', \
                    'their', 'while', 'above', 'both', 'to', 'ours', 'had', \
                    'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',\
                    'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',\
                    'yourselves', 'then', 'that', 'because', 'what', 'over', \
                    'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you',\
                    'herself', 'has', 'just', 'where', 'too', 'only', 'myself',\
                    'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',\
                    'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',\
                    'how', 'further', 'was', 'here', 'than'];
        
        filtered_mails=[]
        pols=[]
        porter = PorterStemmer()   #Stemming to root words 
        for i, mail in enumerate(mails):     #each mail
            if(type(mail)==float):
                continue
            mail = mail.lower()
            mail = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",mail).split()) #remove puncs
            mail = re.sub(r'(.)\1+', r'\1\1', mail)     #remove repeating characterssss

            #mail_words=mail.split()
            stemmed_words = [porter.stem(word) for word in word_tokenize(mail)]
            filtered_words=[]
            
            for word in stemmed_words:
                if (word not in stop_words) and word.isdigit() == False:
                    filtered_words.append(word)
                
            if len(filtered_words)>=2 and len(filtered_words)<=300:
                filtered_mails.append(filtered_words)
                pols.append(polarity.iloc[i])

        return filtered_mails, pols
        
    
    def classifier_dataset(self, filtered_mails, polarity):
        x = []
        y = []
        
        for mail_no, mail in enumerate(filtered_mails):  #cycling through each mail
            mail_len = len(mail)
            for i, word in enumerate(mail):     #cycling through each word
                if word in self.word_index.keys():
                    current=self.word_index[word]        #index of current_word
                else:
                    continue
        
                contexts_a=[]               #indexes of before words
                contexts_b=[]               #indexes of after words
                for j in range(i-self.window, i):
                    if j>=0 and mail[j] in self.word_index.keys():
                        contexts_a.append(self.word_index[mail[j]])
                    else:
                        #contexts_a.append(-1)
                        contexts_a.append(self.voc_size)
                    
                for j in range(i+1, i+1+self.window):
                    if j<mail_len and mail[j] in self.word_index.keys():
                        contexts_b.append(self.word_index[mail[j]])    
                    else:
                        #contexts_b.append(-1)
                        contexts_b.append(self.voc_size)
                
                inp = contexts_a + [current] + contexts_b
                if(polarity[mail_no]==4):     #1-sensitive
                    pol=[1,0]     
                else:
                    pol=[0,1]      #0-non-sensitive
                x.append(inp)
                y.append(pol)

        return x, y  
    
    def test_classifier(self, test_x, test_y):
        length = len(test_x)
        print("No of input data: ", length)        
        
        #graph architecture
        graph=tf.Graph()
        with graph.as_default():
            input_len = self.window*2 + 1
            
            #Predctions for the testing
            look_ups = tf.nn.embedding_lookup(self.w1, test_x)               
            lookup_layer=tf.reshape(look_ups, shape=(-1, input_len*self.n) )        
            linear_layer1= tf.add(tf.matmul(lookup_layer, self.w2), self.b2, name='linear_layer1')
            htanh_layer  = tf.tanh(tf.matmul(linear_layer1, self.w3) + self.b3, name='htanh_layer')
            linear_layer2_s = tf.add(tf.matmul(htanh_layer, self.w4_s), self.b4_s, name='linear_layer2_s')
            output_layer_s = tf.add(tf.matmul(linear_layer2_s, self.w5), self.b5, name='output_layer_s') #softmax loss
            test_prediction = tf.nn.softmax(output_layer_s)                        
        #graph architecture ends####
        
        def accuracy(predictions, labels):
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])
               
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            pred_eval = test_prediction.eval()
            acc = accuracy(pred_eval, test_y)
           
            conf_mat = tf.confusion_matrix(
                            np.argmax(test_y,1),
                            np.argmax(pred_eval,1),
                            num_classes=2,
                        )
            a = conf_mat.eval()
            
            print("\n\t\tPREDICTED_Negative\tPREDICTED_Positive")
            print("TRUE_Negative\t",a[0][0],"\t\t",a[0][1])
            print("TRUE_Positive\t\t",a[1][0],"\t\t",a[1][1])
            recall      = a[1][1]/(a[1][1] + a[1][0])
            precision   = a[1][1]/(a[1][1] + a[0][1])
            F1 = 2 * (precision * recall) / (precision + recall)
            
            print("\nAccuracy: {:.1f}".format(acc))
            print("Recall: ", recall)
            print("Precision: ", precision)
            print("Macro-F1 score: ", F1)
        
#DRIVER CODE
if __name__=='__main__':
    np.random.seed(0)
     
    model = NeuralNet()
    
    print("Reading data...")                             
    data=pd.read_csv('sdata.csv')   #each row should contain: [label, mail]
   
    mails    = data.iloc[750000:850000, 1]   #full
    polarity = data.iloc[750000:850000, 2]   #full
    
    print("Length of testing dataset: ", len(mails))
    
    print("Generating corpus...", time.ctime(time.time()))    
    filtered_mails, polarity= model.create_corpus(mails, polarity)
    
    print("Generating classifier testing_data...", time.ctime(time.time()))
    x, y = model.classifier_dataset(filtered_mails, polarity)   #1 = 196292     #0 = 321109 

    print("Testing classifier...", time.ctime(time.time()))
    model.test_classifier(x, y)      #training classifier neural network
        
    print("PROCESS COMPLETED.")