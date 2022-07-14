# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 14:49:19 2022

@author: kkalinin
"""

import os
import openai
import pandas as pd
import numpy as np
import re
import statistics
from itertools import permutations
import pickle
import re


openai.api_key = os.environ["OPENAI_API_KEY"]
openai.Model.list()

os.chdir('C:/Users/Kirill/Desktop/GPT-3/GitHub')
qdat = pd.read_csv('elites_data_cut.csv')
#DO NOT RUN: qdat = pd.read_csv('elites_data_full.csv') 

#Elites
insertname="The typical member of the Russian elites"
insertphrase=""
comput_results_elites = run_EliteSurvey (qdat, insertname, insertphrase)
pickle.dump(comput_results_elites, open("results_elites.pkl", "wb" )) #save object

#Russians
insertname="an average Russian"
insertphrase=""
comput_results_average = run_EliteSurvey (qdat, insertname, insertphrase)
pickle.dump(comput_results_average, open("results_average.pkl", "wb" )) #save object

#Vladimir Putin
insertname="Vladimir Putin"
insertphrase="Vladimir Putin is the President of Russia."
comput_results_putin = run_EliteSurvey (qdat, insertname, insertphrase)
pickle.dump(comput_results_putin, open("comput_results_putin_add.pkl", "wb" )) #save object

#Alexei Navalny
insertname="Alexei Navalny"
insertphrase="Alexei Navalny is the leader of Russian opposition."
comput_results_navalny = run_EliteSurvey (qdat, insertname, insertphrase)
pickle.dump(comput_results_navalny, open("comput_results_navalny_add.pkl", "wb" )) #save object


############################################
#                FUNCTIONS                 #
############################################

##############FUNCTIONS START###############

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt


def run_EliteSurvey (qdat, insertname="", insertphrase=""):
    gpt3_results = run_GPT3(qdat=qdat, insertname=insertname, insertphrase=insertphrase)
    comput_results = gpt3_results[0]
    orig_results=pd.DataFrame(gpt3_results[1])
    orig_results['Index']=list(orig_results.index.values)
    comput_results['Index']=list(comput_results.index.values)
    merged_dat=orig_results.set_index('Index').join(comput_results.set_index('Index'))

    worded_resp=[]
    
    for i in range(0, merged_dat.shape[0], 1):
        chosenO = merged_dat[['chosen']].loc[i].values.tolist()
        chosenI=-1        
        if(chosenO==['A']):
            chosenI=0
        elif(chosenO==['B']):
            chosenI=1
        elif(chosenO==['C']):
            chosenI=2
        elif(chosenO==['D']):
            chosenI=3
        elif(chosenO==['E']):
            chosenI=4
        elif(chosenO==['F']):
            chosenI=5
        elif(chosenO==['G']):
            chosenI=6
        elif(chosenO==['H']):
            chosenI=7
        elif(chosenO==['I']):
            chosenI=8
        elif(chosenO==['J']):
            chosenI=9
        worded_resp.append(merged_dat[[2]].loc[i].values.tolist()[0][chosenI][3:])
    
    merged_dat['worded_resp'] = worded_resp
    questionN = merged_dat[0].values.tolist()
    questionI = np.unique(questionN).tolist()
    merged_dat['question_wording'] =  [str(i[0]) + ". " + str(i[1]) for i in zip(questionN,  merged_dat[1].values.tolist())]
    
    questionsList={}
    for q in questionI:
        question_wording =  np.unique(merged_dat['question_wording'][merged_dat[0]==q])
        merged_dat_s = merged_dat[merged_dat[0]==q]
        optionI = np.unique(merged_dat_s[['worded_resp']]).tolist()
        optionsList={}
        for l in optionI:
            option_scores = merged_dat_s['chosen_score'][merged_dat_s['worded_resp']==l].values
            mean_scores = statistics.mean(option_scores)
            if(len(option_scores)>1):
                sd_scores = statistics.stdev(option_scores)
            else:
                sd_scores = 0        
            optionsList[l] = [mean_scores, sd_scores]
        questionsList[str(question_wording)]=optionsList
    return(questionsList)


def run_GPT3(qdat, insertname, insertphrase):
    run_lists_questions = create_list_questions(qdat=qdat, insertname=insertname, insertphrase=insertphrase)
    list_questions = run_lists_questions[0]
    extra_questions = run_lists_questions[1]
    data = []

    for question in list_questions:
        gpt3_response = call_GPT3(question)
        data.append(gpt3_response) 
    
    res_dat = pd.DataFrame(data, columns = ['question', 
                                            'answer1','answer2','answer3','answer4', 'answer5', 
                                            'score1','score2','score3','score4', 'score5', 
                                            "chosen", "chosen_score", "id", "model", "object"])
    return(res_dat, extra_questions)


def create_list_questions(qdat, insertname, insertphrase):
    qdat.set_index("Index", inplace=True, drop=False)
    list_of_questions = []
    list_of_extra = []
    for question in range(1, qdat.shape[0]+1, 1):
        print(question)
        newlist = [x for x in qdat.loc[[question]].values.tolist()[0] if pd.isnull(x) == False]
        listopt = ['A. ', 'B. ', 'C. ', 'D. ','E. ','F. ','G. ','H. ','I. ','J. ']
        listoptQ = listopt[0:len(newlist[4:])]
        permopt = newlist[2]
        
        if permopt=='yes' or permopt=='Yes' or permopt==True:
            perm_opts = list(permutations(newlist[4:]))
            perm_opts = [j for i in perm_opts for j in i]
        else:
            perm_opts = newlist[4:]
        
        listoptQf = listoptQ * int(len(perm_opts)/len(listoptQ))
        
        perm_opts = [listoptQf[i] + perm_opts[i] for i in range(0, len(perm_opts), 1)]

        questionW = newlist[3]
        
        if insertname!="" or insertphrase!="":
            questionW = re.sub(r"\[PERSON\]", insertname, questionW)
            questionW = insertphrase + " " + questionW                

        for opt in range(0, len(perm_opts), len(listoptQ)):
            opts=perm_opts[opt:(opt+len(listoptQ))]
                
            output = str(questionW) + '\n' + '\n'.join(opts) + '\n\nAnswer:'
            list_of_questions.append(output)
            
        for opt in range(0, len(perm_opts), len(listoptQ)):
            opts=perm_opts[opt:(opt+len(listoptQ))]
            list_of_extra.append([question, questionW, opts])
           
        print("Number of questions generated is " + str(len(list_of_questions)))
    return(list_of_questions, list_of_extra)


def call_GPT3(prompt):
    kwargs = {"engine":"text-davinci-002", 
              "temperature":0, 
              "max_tokens":1,
              "top_p":1,
              "frequency_penalty":0, 
              "presence_penalty":0, 
              "logprobs":10}
    
    print(prompt)
   
    response = openai.Completion.create(prompt=prompt, **kwargs)
    scores = pd.DataFrame([response["choices"][0]["logprobs"]["top_logprobs"][0]]).T
    scores.columns = ["logprob"]
    scores["%"] = scores["logprob"].apply(lambda x: 100*np.e**x)
    scores = scores.sort_values(by ='%', ascending=False)
    score_names = [re.sub(r"^\s+|\s+$", "", i).strip()  for i in scores.index]
    chosen = [letter for letter in score_names if letter.isalpha()][0]
    
    chosen_score =  scores[['%']].iloc[[ind for ind, letter in enumerate(score_names) if letter.isalpha()][0]].values.tolist()
    
    res = []; res.append(prompt); res.append(score_names); res.append(list(scores["%"])); res.append(chosen);  res.append(chosen_score);  
    res.append(list(pd.DataFrame([response["id"]])[0])); res.append(list(pd.DataFrame([response["model"]])[0]));
    res.append(list(pd.DataFrame([response["object"]])[0]))
    res = flatten(res)
    
    return(res)

###############FUNCTIONS END################
