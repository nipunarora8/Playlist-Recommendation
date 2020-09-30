from flask import Flask, redirect, url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os,cv2
from keras.models import *
from pymongo import MongoClient
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

gender_model=load_model("gender_new.h5")
age_model=load_model("age_best.h5")
emo_model=load_model("expression.h5")

cluster = MongoClient("---------mongodb-atlas-client--------------")
db=cluster['playlist']
collection=db['data']

def predict_age_gen(img_path, age_model,gender_model):

    age_classes=['0-5', '12-17', '18-30', '30-50', '50+', '6-11']
    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img,1.3,5)
    x,y,w,h = faces[0]
    imgg=img[y:y+h,x:x+h]
    imgg=cv2.resize(imgg,(64,64))/255.0
    imgg=imgg.reshape(1,64,64,3)

    pred_gender=gender_model.predict(imgg)
    pred_age=age_model.predict_classes(imgg)
    pred_a=age_classes[pred_age[0]]
    
    if pred_gender[0][0]>0.5:
        pred_g="Male"
    else:
        pred_g="Female"

    return (pred_a,pred_g)

def predict_emo(img_path, emo_model):

    class_to_label = {0 :'Angry', 1 : 'Disgust', 2:'Fear', 3 :'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

    img=cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img,1.3,5)
    x,y,w,h = faces[0]
    imgg=img[y:y+h,x:x+h]
    imgg=cv2.resize(imgg,(48,48))/255.0
    imgg=imgg.reshape(1,48,48,1)

    n_pred=emo_model.predict_classes(imgg)
    output=class_to_label[n_pred[0]]
    return output

@app.route('/predict', methods=['POST','GET'])
def predict():
    
    if request.method == 'POST':
        
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        age,gender=predict_age_gen(file_path,age_model,gender_model)
        emotions=predict_emo(file_path,emo_model)

        result= collection.find({}) 
        df=pd.DataFrame(result)
        df.drop('_id',inplace=True,axis=1)

        drp=[]
        for i in range(len(df)):
            if age not in df.iloc[i]['Age'] or emotions not in df.iloc[i]['Mood'] or gender not in df.iloc[i]['Gender']:
                drp.append(i)
        df.drop(drp,inplace=True)
        df.drop(['Mood','Age','Gender'],axis=1,inplace=True)

        return render_template('pred.html',age=age,gender=gender,emotions=emotions,data=df,file_name=str(f.filename))
	
@app.route('/upload/<filename>')
def upload_img(filename):
    return send_from_directory("uploads", filename)

@app.route('/playlist')
def playlist():

    result= collection.find({}) 
    df=pd.DataFrame(result)
    df.drop('_id',inplace=True,axis=1)
    df.drop(['Mood','Age','Gender'],axis=1,inplace=True)

    return render_template('playlist.html',data=df)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)