#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:58:50 2019
SENSITIVITY CLASSIFICATION - Training
@author: Raj Kumar Shrivastava
"""
import pandas as pd
import numpy as np
import pickle
import time
import random
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import tensorflow as tf

class NeuralNet():
    def __init__(self, config):                           #initializing hyperparameters
        
        self.m = config['m']
        self.classifier_epochs = config['classifier_epochs']
        self.classifier_eta = config['classifier_eta']
        self.window = config['window_size']
        self.mini_batch_size_classifier=config['mini_batch_size_classifier']
        self.k = config['k']
        self.alpha_pred = config['alpha_pred']
        self.word_index = pickle.load(open('word_index.txt','rb'))
        self.skipgrams = np.load('skipgrams.npy')
        self.n = len(self.skipgrams[0])
        self.voc_size = len(self.skipgrams)
        pass
    
    def create_corpus(self, mails, polarity):            #data cleansing
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
        x_noise = []
        y = []

        for mail_no, mail in enumerate(filtered_mails):  #cycling through each mail
            mail_len = len(mail)
            for i, word in enumerate(mail):                 #cycling through each word
                if word in self.word_index.keys():
                    current=self.word_index[word]    #index of current_word
                    noise=current
                else:
                    continue
                
                while(noise==current):
                    noise=random.randint(0, self.voc_size-1)
        
                contexts_a=[]                        #indexes of before words
                contexts_b=[]                        #indexes of after words
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
                noise_inp = contexts_a + [noise] + contexts_b
                if polarity[mail_no]==4: 
                    pol=[1,0]        # 1 = positive
                else:
                    pol=[0,1]        # 0 = negtative
                x.append(inp)
                x_noise.append(noise_inp)
                y.append(pol)

        return x, y, x_noise    
    
    def train_classifier(self, dataset_x, dataset_y, dataset_x_noise):
        temp = list(zip(dataset_x, dataset_y, dataset_x_noise))
        random.shuffle(temp) 
        dataset_x, dataset_y, dataset_x_noise = zip(*temp)
        
        length = len(dataset_x)
        idx1 = int(0.97*length)
        train_x = dataset_x[:idx1]
        train_y = dataset_y[:idx1]
        train_x_noise = dataset_x_noise[:idx1]
        
        valid_x = dataset_x[idx1:]
        valid_y = dataset_y[idx1:]
    
        print("Length of training data: ", len(train_x))
        print("Length of validation data: ", len(valid_x))
        print('Learning rate: ', self.classifier_eta)
        print("Minibatch size: ", self.mini_batch_size_classifier)
        
        #graph architecture
        graph=tf.Graph()
        with graph.as_default():
            tf.set_random_seed(1)
            input_len = self.window*2 + 1
            print("Input neurons: ", input_len)
            X = tf.placeholder(tf.int32, [None, input_len])
            X_noise = tf.placeholder(tf.int32, [None, input_len])
            Y = tf.placeholder(tf.float32, [None, 2])
            Y_c = tf.placeholder(tf.int32, [None,1])
            keep_prob = tf.placeholder(tf.float32)
            
            #self.w1   = tf.Variable(tf.truncated_normal([self.voc_size+1, self.n], mean=0, stddev=0.1), name='w1') # ALTERNATIVE TO TRAINED SKIPGRAMS
            self.w1 = tf.Variable(tf.convert_to_tensor(self.skipgrams, np.float32))
            print("Shape of self.w1", self.w1.get_shape())
            self.w2   = tf.Variable(tf.truncated_normal([input_len*self.n, self.m], mean=0, stddev=0.1), name='w2') #lookup->linear1
            self.b2   = tf.Variable(tf.truncated_normal([self.m], mean=0, stddev=0.1), name='b2')
            self.w3   = tf.Variable(tf.truncated_normal([self.m, self.m], mean=0, stddev=0.1), name='w3')       #linear1->htanh
            self.b3   = tf.Variable(tf.truncated_normal([self.m], mean=0, stddev=0.1), name='b3')
            self.w4_s = tf.Variable(tf.truncated_normal([self.m, 2], mean=0, stddev=0.1), name='w4')    #htanh->linear2_s
            self.b4_s = tf.Variable(tf.truncated_normal([2], mean=0, stddev=0.1), name='b4')
            self.w5   = tf.Variable(tf.truncated_normal([2, 2], mean=0, stddev=0.1), name='w5')     #linear2_s->softmax
            self.b5   = tf.Variable(tf.truncated_normal([2], mean=0, stddev=0.1), name='b5')
            
            self.w4_c = tf.Variable(tf.truncated_normal([self.m, 1], mean=0, stddev=0.1), name='w4_c')    #htanh->linear2_c
            self.b4_c = tf.Variable(tf.truncated_normal([1], mean=0, stddev=0.1), name='b4_c')
    
            look_ups = tf.nn.embedding_lookup(self.w1, X)               
            lookup_layer=tf.reshape(look_ups, shape=(-1, input_len*self.n) )        
            linear_layer1= tf.add(tf.matmul(lookup_layer, self.w2), self.b2, name='linear_layer1')
            htanh_layer  = tf.tanh(tf.matmul(linear_layer1, self.w3) + self.b3, name='htanh_layer')
            #drop_out = tf.nn.dropout(htanh_layer, keep_prob)
            #linear_layer2_s = tf.add(tf.matmul(drop_out, self.w4_s), self.b4_s, name='linear_layer2_s')
            linear_layer2_s = tf.add(tf.matmul(htanh_layer, self.w4_s), self.b4_s, name='linear_layer2_s')
            output_layer_s = tf.add(tf.matmul(linear_layer2_s, self.w5), self.b5, name='output_layer_s') #softmax loss
            cross_entropy_s = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output_layer_s))
            
            #linear_layer2_c = tf.add(tf.matmul(drop_out, self.w4_c), self.b4_c, name='linear_layer2_c')
            linear_layer2_c = tf.add(tf.matmul(htanh_layer, self.w4_c), self.b4_c, name='linear_layer2_c')

            look_ups_noise = tf.nn.embedding_lookup(self.w1, X_noise)
            lookup_layer_noise=tf.reshape(look_ups_noise, shape=(-1, input_len*self.n) )
            linear_layer1_noise = tf.add(tf.matmul(lookup_layer_noise, self.w2), self.b2, name='linear_layer1_noise')
            htanh_layer_noise  = tf.tanh(tf.matmul(linear_layer1_noise, self.w3) + self.b3, name='htanh_layer_noise')
            linear_layer2_c_noise = tf.add(tf.matmul(htanh_layer_noise, self.w4_c), self.b4_c, name='linear_layer2_noise')
                         
            context_score_nce = tf.divide( tf.exp(linear_layer2_c),\
                                          tf.add( tf.exp(linear_layer2_c),\
                                                 tf.scalar_mul(self.k, tf.exp(linear_layer2_c_noise)) )  )
            
            cross_entropy_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_c, logits=context_score_nce))
                    
            
            cross_entropy = tf.add(tf.scalar_mul(self.alpha_pred, cross_entropy_s),\
                                   tf.scalar_mul((1-self.alpha_pred),cross_entropy_c) )
                  
            #train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
            train_step = tf.train.AdamOptimizer(self.classifier_eta).minimize(cross_entropy)
            #train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)
            
            #Predctions for validation dataset
            look_ups_v = tf.nn.embedding_lookup(self.w1, valid_x)               
            lookup_layer_v=tf.reshape(look_ups_v, shape=(-1, input_len*self.n) )        
            linear_layer1_v= tf.add(tf.matmul(lookup_layer_v, self.w2), self.b2)
            htanh_layer_v  = tf.tanh(tf.matmul(linear_layer1_v, self.w3) + self.b3)
            linear_layer2_s_v = tf.add(tf.matmul(htanh_layer_v, self.w4_s), self.b4_s)
            output_layer_s_v = tf.add(tf.matmul(linear_layer2_s_v, self.w5), self.b5) #softmax loss
            valid_prediction = tf.nn.softmax(output_layer_s_v)
            
        #graph architecture end####
        
        def accuracy(predictions, labels):
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                    / predictions.shape[0])           
        
        temp = list(zip(train_x, train_y, train_x_noise))
        random.shuffle(temp) 
        shuffled_x, shuffled_y, shuffled_x_noise = zip(*temp)
        
        train_len = len(shuffled_x)
        mini_batches_x = [ shuffled_x[k:k+self.mini_batch_size_classifier] for k in range(0, train_len, self.mini_batch_size_classifier) ]
        mini_batches_x_noise = [ shuffled_x_noise[k:k+self.mini_batch_size_classifier] for k in range(0, train_len, self.mini_batch_size_classifier) ]
        mini_batches_y = [ shuffled_y[k:k+self.mini_batch_size_classifier] for k in range(0, train_len, self.mini_batch_size_classifier) ]
            
        # train on mini batches
        print("Training started at ", time.ctime(time.time()) )
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epo in range(self.classifier_epochs):
                loss_sum = 0
                for mini_count in range(len(mini_batches_x)):
                    batch_x = mini_batches_x[mini_count]
                    batch_x_noise = mini_batches_x_noise[mini_count]
                    batch_y = mini_batches_y[mini_count]  
    
                    one1 = np.ones(len(batch_y))
                    one1 = one1.reshape(len(one1),1)
                    
                    feed_dict={X: batch_x, X_noise: batch_x_noise, Y: batch_y, Y_c:one1, keep_prob:0.5}
                    _, mini_loss = sess.run([train_step,cross_entropy], feed_dict)
                    loss_sum += mini_loss   
                    
                avg_loss = loss_sum/len(mini_batches_x)
                print("\nEpoch", epo+1, " completed at ",time.ctime(time.time()), " | Epoch Loss = ", avg_loss)
                print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_y)))
                
            print("Classifier training completed at ", time.ctime(time.time()) )
            
            print("Saving model...")
            np.save('w1.npy',self.w1.eval())
            np.save('w2.npy',self.w2.eval())
            np.save('b2.npy',self.b2.eval())
            np.save('w3.npy',self.w3.eval())
            np.save('b3.npy',self.b3.eval())
            np.save('w4_s.npy',self.w4_s.eval())
            np.save('b4_s.npy',self.b4_s.eval())
            np.save('w5.npy',self.w5.eval())
            np.save('b5.npy',self.b5.eval())
        
#DRIVER CODE
if __name__=='__main__':
    np.random.seed(0)
    config={'m':20, 'window_size':7, 'mini_batch_size_classifier':256, 'classifier_epochs':6, \
            'classifier_eta':0.0003, 'alpha_pred':0.7, 'k':32}   #HYPER-PARAMETERS 
     
    model = NeuralNet(config)
    
    print("Reading data...")
    data=pd.read_csv('sdata.csv')   #each row contains: [label, message]
   
    polarity = data.iloc[:750000, 2].append(data.iloc[850000:,2])   #full
    mails    = data.iloc[:750000, 1].append(data.iloc[850000:,1])   #full
    
    print("Length of training dataset: ", len(mails))
    
    print("Generating corpus...", time.ctime(time.time()))    
    filtered_mails, polarity= model.create_corpus(mails, polarity)
    
    print("Generating classifier training_data...", time.ctime(time.time()))
    x, y, x_noise = model.classifier_dataset(filtered_mails, polarity)   

    print("Training classifier...", time.ctime(time.time()))
    model.train_classifier(x, y, x_noise)      #training classifier neural network
        
    print("PROCESS COMPLETED.")