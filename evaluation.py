import matplotlib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score
import tensorflow.keras as tfk
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from flask import Flask, request, url_for, flash, redirect, render_template, send_file, Response
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import pre_processing
from pre_processing import cleanup
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
matplotlib.pyplot.switch_backend('Agg')
matplotlib.use('Agg')
import os
import pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
meta_path= 'meta_vat'
no_row=608
# loading models
MT_synonym= load_model ( 'models/MT_synonym')
MT_unlabel=load_model ( 'models/MT_unlabel')
MT_dropout=load_model ( 'models/MT_dropout')
MT_synanddrop=load_model ( 'models/MT_synanddrop')



# model_vat is supposed to be the last model zip file I sent
# call test_vat()
def prec_rec_f1score(y_true, x_test, model,item) :
    bce = tf.keras.metrics.BinaryCrossentropy()
    # print(model.summary() )
    y_hat = model.predict(x_test)
    y_pred = (np.greater_equal( y_hat, 0.505 )).astype ( int )
    # for psuedo labelling and Vat technique
    # print(item+'********')
    # y_hat= np.max(y_hat,axis=1)# this one for calculating binary loss
    # print(y_hat)
    # print(y_pred)
    pr_re_f1score_perclass = precision_recall_fscore_support (y_true, y_pred, average=None )
    pr_re_f1score_average = precision_recall_fscore_support ( y_true, y_pred, average='micro' )
    precision = precision_score( y_true, y_pred, average=None )
    recall = recall_score( y_true, y_pred, average=None )
    accuracy = accuracy_score( y_true, y_pred )
    f1_score = f1(y_true, y_pred )
    # per class
    precision_true = pr_re_f1score_perclass[0][1]
    precision_fake = pr_re_f1score_perclass[0][0]
    recall_true = pr_re_f1score_perclass[1][1]
    recall_fake = pr_re_f1score_perclass[1][0]
    f1score_true = pr_re_f1score_perclass[2][1]
    f1score_fake = pr_re_f1score_perclass[2][0]

    fpr_, tpr_, _ = roc_curve( y_true, y_pred)
    binary_loss = bce ( y_true, y_hat ).numpy ()
    return accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, binary_loss,fpr_, tpr_, y_pred
# tokenizer function

def tokenize(x_test):
    token_path = 'tokenizer'
    with open(token_path+'/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    maxlen=100
    x_test_seq= tokenizer.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=maxlen,padding='post')
    return x_test

def model_evaluation(df) :
    folder_name = os.listdir ( 'models' )
    folder_name.pop( 0 )
    x_test = (df['Article'].astype ( str )).values
    df['Label'] = df['Label'].astype (int)
    y_test = (df['Label'].astype ( int )).values
    x_test = tokenize (x_test)
    # creating dataframe
    report_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision_True', 'Precision_Fake', 'Recall_True',
                                        'Recall_Fake', 'F1_Score_True', 'F1_Score_Fake', 'Classification_Loss'] )
    predict_df= pd.DataFrame()
    predict_df= df.copy()
    pyplot.Figure()
    for items in folder_name :
        print(items)
        model = load_model ( 'models/' + items)
        accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, \
        f1score_fake, binary_loss, fpr, tpr,y_pred = prec_rec_f1score( y_test, x_test, model,items)
        predict_df[items]= y_pred
        # print('before pyplot')
        pyplot.plot(fpr, tpr, linestyle='--', label=items )
        report_df=report_df.append( {'Model' : items, 'Accuracy' : accuracy, 'Precision_True' : precision_true,
                                        'Precision_Fake' : precision_fake, 'Recall_True' : recall_true,
                                        'Recall_Fake' : recall_fake,
                                        'F1_Score_True' : f1score_true, 'F1_Score_Fake' : f1score_fake,
                                        'Classification_Loss' : binary_loss}, ignore_index=True )
    # print('outside for loop')
    pyplot.xlabel('False Positive Rate' )
    pyplot.ylabel('True Positive Rate' )
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.savefig('static/images/roccurve_overall.png', format='png', dpi=1200 )
    # pyplot.show()
    plt.clf()
    return report_df, predict_df
def model_prediction(article):
    # text processing
    article = cleanup(article)
    print(article)
    # arti_ar=np.array([article])
    # print(arti_ar)
    # Tokenization
    article_seq=tokenize([article])
    print(article_seq,np.shape(article_seq))
    predict_df = pd.DataFrame()
    predict_df['Article']= [article]
    print(MT_synonym.predict(article_seq))
    y_hat=((MT_synonym.predict(article_seq)).round())
    predict_df['MT_synonym'] = (np.greater_equal ( y_hat, 0.505 ))
    y_hat = ((MT_dropout.predict ( article_seq )).round ())
    predict_df['MT_dropout'] = (np.greater_equal ( y_hat, 0.505 ))
    y_hat = ((MT_synanddrop.predict ( article_seq )).round ())
    predict_df['MT_synanddrop'] = (np.greater_equal ( y_hat, 0.505 ))
    y_hat = ((MT_unlabel.predict ( article_seq )).round ())
    predict_df['MT_Unlabel'] = (np.greater_equal ( y_hat, 0.505 ))
    predict_df.replace({False:'Fake'})
    return predict_df
def model_prediction_df(df):
    for i in range(len(df)):
        df.iloc[i, 0] = cleanup(df.iloc[i,0])

    x_test_seq= tokenize(df.iloc[:,0])
    # print(x_test_seq , np.shape(x_test_seq))
    predict_df = pd.DataFrame ()
    predict_df['Article'] = df['Article']
    predict_df['MT_synonym'] =  ( (MT_synonym.predict ( x_test_seq )).round () ).astype(bool)
    predict_df['MT_dropout'] =  ( (MT_dropout.predict ( x_test_seq )).round () ).astype(bool)
    predict_df['MT_Unlabel'] = ( (MT_unlabel.predict ( x_test_seq )).round () ).astype(bool)
    predict_df['MT_Synanddrop'] = ( (MT_synanddrop.predict ( x_test_seq )).round () ).astype(bool)
    predict_df.replace( {False : 'Fake'} )
    return predict_df







# if __name__ == '__main__' :
#     article="this is sparta trump bomb politics news the world is not enough"
#     print(model_prediction(article))
#     df = pd.read_csv ( 'templates/x_test.csv', index_col=0 )
#     x_test = (df['Article'].astype ( str )).values
#     df['Label'] = df['Label'].astype ( int )

    # y_test = (df['Label'].astype ( int )).values
    # x_test = tokenize(x_test)
    # folder_name = os.listdir ( 'models' )
    # folder_name.pop ( 0 )
    # for items in folder_name :
    #     model = load_model( 'models/' + items)
    #     y_pred= model.predict(x_test)
    #     fpr_, tpr_, _ = roc_curve( y_test, y_pred )
    #     pyplot.plot ( fpr_, tpr_, linestyle='--', label=items)
    # pyplot.xlabel ( 'False Positive Rate' )
    # pyplot.ylabel ( 'True Positive Rate' )
    # # show the legend
    # pyplot.legend ()
    # # show the plot
    # pyplot.savefig ( 'static/images/roccurve_overall.eps', format='eps', dpi=1200 )
    # pyplot.show ()