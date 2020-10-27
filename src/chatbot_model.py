import pandas as pd
import numpy as np
import nltk
import pickle
nltk.download('wordnet')
from sklearn.feature_extraction import text
from joblib import load
from nltk.tokenize import word_tokenize
import re
from tkinter import Entry
from tkinter import *

from datetime import datetime

from liste import *
from timetable import * 

count_vec = load("../models/count_vec.joblib")


Jours = {0:"Lundi", 1:"Mardi", 2:"Mercredi", 3:"Jeudi", 4:"Vendredi", 5:"Samedi", 6:"Dimanche"}


def preprocessing_sentence(sentence):

    # Apply CountVectorizer 
    series = pd.Series(sentence)
    transformed_sentence = count_vec.transform(series)
    return transformed_sentence


model = load("../models/model_LogReg.joblib")


def predict_class(input_sentence):

    transformed_sentence = preprocessing_sentence(input_sentence)
    pred = model.predict(transformed_sentence)
    return pred[0]


def getResponse(pred,input_sentence):

    # Si impression
    if pred == 1:
        number_pages = re.findall(r'\s([0-9]+)\s',input_sentence)
        doc_name     = re.findall(r'[D-d]oc[0-9]*\s', input_sentence)

        return f"I will print the {doc_name[0]} with {number_pages[0]} pages"
    # Autres
    elif pred == 0:
        return "I don't understand what you mean. Please, resend another request."


def time():
    now = datetime.now()
    # The correct format HH/MM/SS
    return now.strftime("%H:%M:%S")

def heure():
    t = datetime.now()
    return t.hour*h + t.minute*m + t.second * s



def send():
    global EntryBox
    global ChatBox
    
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':

        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, '[' + time() + '] ' "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 ))

        number_pages = re.findall(r'\s([0-9]+)\s',msg)
        doc_name     = re.findall(r'[D-d]oc[0-9]*\s', msg)

        if (len(number_pages) == 0):
            res = "argument missing: number of pages"
        elif (len(doc_name) == 0):
            res = "argument missing: name of document"
        else:
            pred = predict_class(msg)
            res = getResponse(pred,msg)

        ChatBox.insert(END, '[' + time() + '] ' "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
        
        jour = Jours[datetime.today().weekday()]
        nouvelle_tache(heure(), int(number_pages[0]) , doc_name[0], jour)

def see():
    first = getList()
    curr  = first
    
    init_timetable(first)
    
    
    
def init_interface():
    global EntryBox
    global ChatBox
    
    root = Tk()
    root.title("Chatbot")
    root.geometry("400x500")
    root.resizable(width=FALSE, height=FALSE)

    #Create Chat window
    ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)

    ChatBox.config(state=DISABLED)

    #Bind scrollbar to Chat window
    scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
    ChatBox['yscrollcommand'] = scrollbar.set

    #Create Button to send message
    SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                        command= send )

    SeeButton = Button(root, font=("Verdana",12,'bold'), text="See", width="12", height=5,
                        bd=0, bg="#fcdb03", activebackground="#3c9d9b",fg='#000000',
                        command= see )

    #Create the box to enter message
    EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")


    #Place all components on the screen
    scrollbar.place(x=376,y=6, height=386)
    ChatBox.place(x=6,y=6, height=386, width=370)
    EntryBox.place(x=128, y=401, height=90, width=265)
    SendButton.place(x=6, y=401, height=45)
    SeeButton.place( x=6, y=401+45, height=45)

    root.mainloop()
