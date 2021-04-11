from flask import Flask,render_template,request, redirect, make_response, url_for, session,Response
import pickle
import pandas as pd
import numpy as np
from rake_nltk import Rake
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


app = Flask(__name__)   

###     parameters     ###

user_msg = []
bot_msg = []
user_symp = []
disease = []

###      routing section starts     ###
@app.route('/',methods=['POST','GET'])    
def home():
    if request.method=='POST':
        user_name= request.form['user_name']
        user_age= request.form['user_age']
        resp=make_response(redirect('/messager'))
        resp.set_cookie('User_name',user_name) 
        resp.set_cookie('user_age',user_age)
        resp.set_cookie('diagnose','False') 
        print(user_name,user_age)
        return resp
    else:
        return render_template('index.html');  

@app.route('/feedback')    
def feedback():
    return render_template('feedback.html');  


@app.route('/messager',methods =["GET","POST"])    

def messager():
    diagnose = request.cookies.get('diagnose')
    diagnose_true = request.cookies.get('diagnose_true')
    voice_msg = None
    if diagnose == 'False':
        if request.method == "POST":
            current_msg = request.form['msg']
            add_user_msg(current_msg)
            print(user_msg)
            
            if 'Start diagnose' in current_msg:
                add_bot_msg('Started')
                length = max(len(user_msg),len(bot_msg))     
                voice_msg= bot_msg[len(bot_msg)-1] 
                resp=make_response(render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length, voice_msg= voice_msg))
                resp.set_cookie('diagnose', '', expires=0)
                resp.set_cookie('diagnose_true','True')
                return resp
            
            reply = chat_resp(current_msg)
            add_bot_msg(reply)
            voice_msg = None
            length = max(len(user_msg),len(bot_msg))     
            voice_msg= bot_msg[len(bot_msg)-1] 
            return render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length,voice_msg =voice_msg)
        else:
            voice_msg = None
            length = max(len(user_msg),len(bot_msg))     
            voice_msg= bot_msg[len(bot_msg)-1] 
            return render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length,voice_msg =voice_msg)
    
    elif diagnose_true == 'True':
        if request.method == "POST":
            current_msg = request.form['msg']
            add_user_msg(current_msg)
            print(user_msg)
            
            if 'Stop diagnose' in current_msg:
                add_bot_msg('Stopped')
                length = max(len(user_msg),len(bot_msg))     
                voice_msg= bot_msg[len(bot_msg)-1] 
                resp=make_response(render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length, voice_msg= voice_msg))
                resp.set_cookie('diagnose_true', '', expires=0)
                resp.set_cookie('diagnose','False')
                user_symp.clear()
                return resp
            
            val, val_prob = predict(current_msg)
            if val_prob < 0.956:
                resp = random_resp(symp_error)
                add_bot_msg(resp)
                length = max(len(user_msg),len(bot_msg))     
                voice_msg= bot_msg[len(bot_msg)-1] 
                return render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length, voice_msg= voice_msg)
            else:
                val_prob = val_prob * 100
                resp = "I predicted you might "+ str(val) + ", with probability of "+ '%.2f'%val_prob
                add_bot_msg(resp)
                disease.append(val)
                voice_msg = None
                length = max(len(user_msg),len(bot_msg))     
                voice_msg= bot_msg[len(bot_msg)-1] 
                render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length,voice_msg =voice_msg)
                val = ''.join(val)
                val = des[des['Disease'].str.lower() == val]
                add_bot_msg('Description about predicted disease')
                text = convert(val.Description)
                add_bot_msg(text)
                voice_msg = None
                length = max(len(user_msg),len(bot_msg))     
                voice_msg= bot_msg[len(bot_msg)-1] 
                render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length,voice_msg =voice_msg)

                
                return render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length,voice_msg =voice_msg)
        
        else:
            voice_msg = None
            length = max(len(user_msg),len(bot_msg))     
            voice_msg= bot_msg[len(bot_msg)-1]
            return render_template('messager.html',bot_msg = bot_msg, user_msg=user_msg, length= length, voice_msg= voice_msg) 
   


###    routing section ends    ###
if __name__ =='__main__':   
    
    
###  importing csv files   ###
    greet_ques = pd.read_csv('response//greet_ques_db.csv')
    intro = pd.read_csv('response//intro_db.csv')
    symp_greet = pd.read_csv('response//symp_greet_db.csv')
    symp = pd.read_csv('response//symp_db.csv')
    symp_error = pd.read_csv('response//symp_error_db.csv')
    des = pd.read_csv('csv/symptom_Description.csv')
    prec = pd.read_csv('csv/symptom_precaution.csv')
    sev = pd.read_csv('csv/Symptom-severity.csv')

###     function sections starts    ###
    with open("CPS.pkl", 'rb') as file:  
        CPS_Model = pickle.load(file)
    
    #creating a new chatbot
    chatbot = ChatBot("Chatterbot",storage_adapter="chatterbot.storage.SQLStorageAdapter")
    trainer = ChatterBotCorpusTrainer(chatbot)
    trainer.train("chatterbot.corpus.english")
    
    def chat_resp(ques):
        response = chatbot.get_response(ques)
        return (response)
    
    class RakeImpl:
        def __init__(self, text):
            self.text = text
            self.rake = Rake()
        def getKeywords(self):
            self.rake.extract_keywords_from_text(self.text)
            return self.rake.get_ranked_phrases()
        
    def random_resp(df):
        text = df['0'].sample(n=1)
        text = np.array(text)
        return (''.join(text))
    
    def convert(text):
        return (''.join(str(v) for v in text))
     
    def preprocess(user_text):
        
        rakeImpl = RakeImpl(user_text)
        user_text = rakeImpl.getKeywords()[:20]
        user_text = ''.join(user_text)
        print(user_text)
        return user_text
            
    
    def predict (symp):
        user_symp.append(symp)
        symp = ''.join(user_symp)
        symp = preprocess(symp)
        y_pred = CPS_Model.predict_proba([symp])
        pred = CPS_Model.predict([symp])
        return pred,y_pred.max()
    
   
    def add_user_msg(msg):
        user_msg.append(msg)

    def add_bot_msg(msg):
        bot_msg.append(msg)

###     response generation    ###

    add_bot_msg(random_resp(intro))
        


###     flask functions    ###
    app.run(debug = True)
    