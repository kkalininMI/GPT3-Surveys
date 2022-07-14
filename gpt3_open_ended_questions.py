# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 16:08:38 2022

@author: kkalinin
"""

from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
import os
import pandas as pd
import re
import openai
import pickle
from statistics import mean, stdev
import ipdb
import time

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.Model.list()

kwargs = {"model":"babbage:ft-stanford-university-2022-06-20-08-29-24", 
          "temperature":0, 
          "max_tokens":100,  
          "top_p":1,  
          "frequency_penalty":1, 
          "presence_penalty":1, 
          "best_of":10,
          "stop": ["\n", "<|endoftext|>"]}   


os.chdir('C:/Users/Kirill/Desktop/GPT-3/GitHub')
qdat = pd.read_csv('elites_data_open.csv') 


#Vladimir Putin
insertyear="Before 2014,"
insertname="Vladimir Putin"
insertphrase=""
putin_before2014 = generate_openQ (qdat, insertyear, insertname, insertphrase, kwargs)
putin_before2014_var = AnalyzeGeneratedResponse(putin_before2014)
additional_info=["Similarity: " + i[0][2] +"  " + str(round(max(i[0][0])*100, 0)) + "  Sentiment:" + i[1][0] +"  " + str(round(max(i[1][1]), 0)) for i in putin_before2014_var]

insertyear="After 2020,"
putin_after2020 = generate_openQ (qdat, insertyear, insertname, insertphrase, kwargs)
putin_after2020_var = AnalyzeGeneratedResponse(putin_after2020)
additional_info=["Similarity: " + i[0][2] +"  " + str(round(max(i[0][0])*100, 0)) + "  Sentiment:" + i[1][0] +"  " + str(round(max(i[1][1]), 0)) for i in putin_after2020_var]


#Alexei Navalny
insertyear="Before 2014,"
insertname="Alexei Navalny"
insertphrase=""
navalny_before2014 = generate_openQ (qdat, insertyear, insertname, insertphrase, kwargs)
navalny_before2014_var = AnalyzeGeneratedResponse(navalny_before2014)
additional_info=["Similarity: " + i[0][2] +"  " + str(round(max(i[0][0])*100, 0)) + "  Sentiment:" + i[1][0] +"  " + str(round(max(i[1][1]), 0)) for i in navalny_before2014_var]

insertyear="After 2020,"
navalny_after2020 = generate_openQ (qdat=qdat, 
                                    insertyear=insertyear, 
                                    insertname=insertname, 
                                    insertphrase=insertphrase, kwargs=kwargs)
navalny_after2020_var = AnalyzeGeneratedResponse(navalny_after2020)
additional_info=["Similarity: " + i[0][2] +"  " + str(round(max(i[0][0])*100, 0)) + "  Sentiment:" + i[1][0] +"  " + str(round(max(i[1][1]), 0)) for i in navalny_after2020_var]

#Save data
dict_process={}
dict_process['putin_before2014'] = putin_before2014
dict_process['putin_before2014_var'] = putin_before2014_var
dict_process['putin_after2020'] = putin_after2020
dict_process['putin_after2020_var'] = putin_after2020_var
dict_process['navalny_before2014'] = navalny_before2014
dict_process['navalny_before2014_var'] = navalny_before2014_var
dict_process['navalny_after2020'] = navalny_after2020
dict_process['navalny_after2020_var'] = navalny_after2020_var
pickle.dump(dict_process, open("open_ended_generated_res.pkl", "wb" )) #save object


############################################
#                FUNCTIONS                 #
############################################

#############FUNCTIONS START################

########post-generation processing##########
def AnalyzeGeneratedResponse(var):
    general_res = []
    for g in range(0, len(var),1):
        response_text = var[g][1]
        perm_opts = var[g][2]
        similarity = similar_option(perm_opts, response_text)
        sentiment = classify(response_text)
        general_res.append([similarity, sentiment])
    return general_res
 
def classify(response_text):
    response = openai.Completion.create(
              model="text-davinci-002",
              prompt= "This is a sentiment classifier. " +"Text:' " + response_text +" Sentiment:",
              temperature=0,
              max_tokens=60,
              top_p=1.0,
              frequency_penalty=0.0,
              presence_penalty=0.0,
              logprobs=10)

    scores = pd.DataFrame([response["choices"][0]["logprobs"]["top_logprobs"][0]]).T
    scores.columns = ["logprob"]
    scores["%"] = scores["logprob"].apply(lambda x: 100*np.e**x)
    scores = scores.sort_values(by ='%', ascending=False)
    score_names = [re.sub(r"^\s+|\s+$", "", i).strip()  for i in scores.index]

    chosen = score_names[0]

    pos=0; neg=0; neu=0
    scores['Index'] = [i.strip() for i in list(scores.index.values)]
    for i in range (0, scores.shape[0], 1):
        row_n = scores.iloc[[i]]
        ind = list(row_n['Index'])[0]
        val = list(row_n["%"])[0]
        if ind=="Positive" or ind=="positive" or ind=="Pos" or ind=="pos":
            pos += val
        if ind=="Negative" or ind=="negative" or ind=="Neg" or ind=="neg":
            neg += val
    neu = 100 - (neg + pos)
    res = [chosen, [pos, neg, neu]]
    return res
    
def similar_option(perm_opts, response_text):
    embedding = get_embedding(response_text)    
    emb_list = [get_embedding(x) for x in perm_opts]
    similarities = [cosine_similarity(x, embedding) for x in emb_list]
    index_max = np.argmax(similarities)
    sim_opt = perm_opts[index_max]
    mean_res = mean(similarities)
    sd_res = stdev(similarities)
    res = [similarities, response_text, sim_opt, mean_res, sd_res]
    return res

def get_embedding(text, model="text-similarity-babbage-001"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

####open-ended text generation functions####
def generate_openQ (qdat, insertyear, insertname, insertphrase, kwargs):

    qdat.set_index("Index", inplace=True, drop=False)
    results = []
    
    for question in range(1, qdat.shape[0]+1, 1):
        print(question)
        newlist = [x for x in qdat.loc[[question]].values.tolist()[0] if pd.isnull(x) == False]
        questionW = newlist[3]
                
        if insertname!="" or insertphrase!="":
            questionW = re.sub(r"\[PERSON\]", insertname, questionW)
            questionW = re.sub(r"\[YEAR\]", insertyear, questionW)
            questionW = insertphrase + " " + questionW                
            output = str(questionW)
            
        perm_opts = newlist[4:]
        
        print(output)
        
        response = openai.Completion.create(prompt=output, **kwargs)
        response_text =  response['choices'][0]['text']
        res = [output, response_text, perm_opts]
        results.append(res)
    return results

###############FUNCTIONS END################