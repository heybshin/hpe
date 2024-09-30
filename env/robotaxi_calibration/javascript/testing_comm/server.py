 # docs @ http://flask.pocoo.org/docs/1.0/quickstart/

from flask import Flask, jsonify, request, render_template
import requests as req
import subprocess
import mysql.connector
import os
import random
import numpy as np
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'storage/'

@app.route('/RL', methods=['GET', 'POST'])
def learning():

    # POST request
    if request.method == 'POST':
        # to debug
        print('Incoming..')
        print(request.get_json())  # parse as JSON

        # Javascript has returned a json. parse through

        # Probably user input, video/audio raw data

        # RL/Pytorch

        # Process outputs back into correct states/actions (JSON)

        #return
        return 'OK', 200

    # GET request
    else:
        message = {'greeting':'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers

@app.route('/finish', methods=['POST'])
def finish():

    # Store the video file recieved
    if request.method == 'POST':

        # Connect to the localhost database
        mydb = mysql.connector.connect(
            host="localhost",
            db="robotaxi",
            user="root",
            password="Mturk$35@"
        )

        # Get the correct data to send to database
        cursor = mydb.cursor()
        sql_insert_blob_query = """ INSERT INTO player (player_id, video) VALUES (%s, %s)"""

        player_id = random.randint(10000, 99999)
        video_file = request.files['video-blob']
        #insert_blob_tuple = (player_id, video_file)
        #result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        video_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'testing_audio.webm'))
        
        #mydb.commit()

        #with open ('testing.webm', 'wb') as f_output:
            #f_output.write(video_file.stream)

        # if the connection exists, end the connection
        if mydb.is_connected():
            cursor.close()
            mydb.close()
            print("Connection to database closed")
        
        return 'finished?'
    else:
        return "ERROR: Only POST requests allowed for /finish"

    
@app.route('/', methods=['GET', 'POST'])
def page():

    #Fire up the javascript page
    return render_template('index.html')
