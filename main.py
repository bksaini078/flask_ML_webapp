#Codeby: Bhupender Kumar Saini
import os
import os.path
import os.path
from os import path

import numpy as np
import pandas as pd
from flask import Flask, request, url_for, redirect, render_template, flash
from flask_ngrok import run_with_ngrok
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import flask_wtf
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired
from evaluation import model_evaluation, model_prediction,model_prediction_df

# uploading procedure
UPLOAD_FOLDER = 'templates'
no_row=608
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# socketio = SocketIO(app)
# run_with_ngrok(app)  # startsS ngrok when the app is run
# loading model
@app.route('/action', methods=['GET', 'POST'])
def action():
    if request.method == 'POST':
        return redirect(url_for('evaluation'))
    df = pd.read_csv(UPLOAD_FOLDER + '/x_test.csv', index_col=0)
    if 'Label' in df.columns.tolist() :
        df['Label'] = df['Label'].astype(int)
        return """<h1> Uploaded News details, click next for evaluations </h1> 
        <form method=post enctype=multipart/form-data><input type=submit value=Next>
        </form>"""+ df[:no_row].to_html()
    if df.columns.tolist():
        predict= model_prediction_df(df)
        return render_template('simple1.html',
                               tables1=[predict.to_html(classes='data1')])


    # return send_file('/content/templates/shiva.jpg', mimetype='image/jpeg')
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        form= request.form
        article = form.get ( 'article' )
        print(form)
        file = request.files['xfile']
        print(file)
        filename, file_extension = os.path.splitext ( file.filename )
        print(file_extension)
        if file and file_extension =='.csv':
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'x_test.csv'))
            return redirect(url_for('action'))
        if file and file_extension=='.txt':
            file.save( os.path.join ( app.config['UPLOAD_FOLDER'], 'x_test.txt' ))
            file1 = open(UPLOAD_FOLDER+"/x_test.txt", "r" )
            article=file1.read()
            print(article)
            print(type(article))
            predict_txt= model_prediction(article)
            return render_template('simple1.html',tables1=[predict_txt.to_html(classes='data1')])
        if article:
            predict= model_prediction(article)
            return render_template('simple1.html',
                               tables1=[predict.to_html(classes='data1')])
        else:
            return """<h2> Uploaded files are not supported rightnow, Supported formats are .csv and .txt.</h2>"""
    return render_template('index.html')


@app.route('/evaluation')
def evaluation():
    if path.exists(UPLOAD_FOLDER + '/x_test.csv'):
        df = pd.read_csv(UPLOAD_FOLDER + '/x_test.csv', index_col=0)
        df= df[:no_row].copy()
        true_news_count = np.count_nonzero (df['Label'])
        fake_news_count = len (df ) - true_news_count
        # df_r= df[:5]
        df_r,df_p = model_evaluation(df)
        df_r = df_r.sort_values(by=['Accuracy'], ascending=False).reset_index(drop=True)
        # print(df_r.to_html())
        return render_template('simple.html', tables=[df_r.to_html(classes='data')], titles=df_r.columns.values,
                               True_news= true_news_count,Fake_news=fake_news_count, tables1=[df_p.to_html(classes='data1')]) #'''<h>this is testing phase 1''' + df_r.to_html()
    return 'Test file does not exist'


if __name__ == '__main__':
    # socketio.run(app)
    app.run()
