# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:56:37 2023

@author: Kirill
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
import math

#The "Survey of Russian Elites" dataset from 1993 to 2020 (ICPSR 3724), 
#can be accessed at https://www.icpsr.umich.edu/web/ICPSR/studies/3724.

dat = pd.read_spss("ICPSR_data.sav")
dat2020 = dat.loc[dat['YEAR'] == 2020]

################################################################
# Example:                                                     #
# Build socio-demographic variables: age, elite group, gender  #
################################################################
def recode_variable_elites(x):
    mapping = {'(1995 to 2020) Private Business': 'nonstate elites', 
    '(1995 to 2020) State-Owned Enterprises': 'nonstate elites',
    'Media': 'nonstate_elites',
    'Science/Education': 'state elites',
    'Military/Security Agencies': 'state elites',
    '(1999 to 2020) Executive Branch/Ministries': 'state elites',
    '(1999 to 2020) Legislative Branch (those involved in foreign policy)': 'state elites'}    
    y = [mapping[val] for val in x]
    return y

def recode_variable_sex(x):
    mapping = {'Male': 'male', 
    'Female': 'female'}    
    y = [mapping[val] for val in x]
    return y

dat2020["age_group"] = pd.cut(2020-np.array(dat2020['DOB']), 2, include_lowest=True, labels=["young", "old"])
dat2020["elite_group"] = recode_variable_elites(dat2020["WORKGROUP"]); dat2020["elite_group"] = dat2020["elite_group"].astype('category')
dat2020["sex_group"] = recode_variable_sex(dat2020["SEX"]); dat2020["sex_group"] = dat2020["sex_group"].astype('category')

#Q1  The Limits of National Interests (FPNATINT)
#Question: [PERSON] thinks that 
#A. The national interests of Russia for the most part should be limited to its current territory. 
#B. The national interests of Russia for the most part should extend beyond its current territory.

def recode_variable_FPNATINT(x):
    mapping = {"The nat'l interests of Russia for the most part (1993-2016: extend) (2020: should extend) beyond its current territory.": 'The national interests of Russia for the most part should extend beyond its current territory.', 
    'The national interests of Russia for the most part  should be limited to its current territory.': 'The national interests of Russia for the most part should be limited to its current territory.'}    
    default_value = np.nan  # Specify the default value to use for missing keys
    y = [mapping.get(val, default_value) for val in x]
    return y

dat2020["FPNATINT"] = recode_variable_FPNATINT(dat2020["FPNATINT"]);# dat2020["natinterests"] = dat2020["natinterests"].astype('category')
freq_table_FPNATINT = pd.crosstab([dat2020['age_group'], dat2020['elite_group'], dat2020['sex_group']],  columns=dat2020['FPNATINT'], normalize='index')
demprompt = "In {0} {1} member of Russian elite who belongs to {2} and {3} thinks that"
gpt_FPNATINT = run_GPT3(freq_table_FPNATINT, 
                     demprompt = demprompt,
                     yearinfo = "2020",
                     questtext = "",
                     permopt = "yes")

#########################################
#            MAIN FUNCTIONS             #
#########################################
def prompts_frequency_tables(freq_table, 
                             demprompt = "",
                             yearinfo = "", 
                             questtext = "",
                             permopt = "yes"):
    
    list_of_questions = []
    list_of_extra = []
        
    insertions =  [(yearinfo,) + i  for i in list(freq_table.index)]    
    dem_list = [demprompt.format(*i) for i in insertions]
    question_list = [i + questtext for i in dem_list]
    
    options = list(freq_table.columns); n_perm_opts = math.factorial(len(options))
    listopt = ['A. ', 'B. ', 'C. ', 'D. ','E. ','F. ','G. ','H. ','I. ','J. ']
    listoptQ = listopt[0:len(options)]
    
    for question in range(0, len(question_list), 1):
        print(question)
        if permopt=='yes' or permopt=='Yes' or permopt==True:
            perm_opts = list(permutations(options))
            perm_opts = [j for i in perm_opts for j in i]
        else:
            perm_opts = options
        
        listoptQf = listoptQ * int(len(perm_opts)/len(listoptQ))
        perm_opts = [listoptQf[i] + perm_opts[i] for i in range(0, len(perm_opts), 1)]
        questionW = question_list[question]
        
        for opt in range(0, len(perm_opts), len(listoptQ)):
            opts = perm_opts[opt:(opt + len(listoptQ))]
            output = str(questionW) + '\n' + '\n'.join(opts) + '\n\nAnswer:'
            list_of_questions.append(output)
        
        for opt in range(0, len(perm_opts), len(listoptQ)):
            opts=perm_opts[opt:(opt+len(listoptQ))]
            list_of_extra.append([question, questionW, opts])
            
    print("Number of questions generated is " + str(len(list_of_questions)))
    
    return(list_of_questions, list_of_extra, n_perm_opts)
    

def run_GPT3(freq_table, demprompt, yearinfo, questtext, permopt):
    
    run_lists_questions = prompts_frequency_tables(freq_table = freq_table, 
                                                   demprompt = demprompt, 
                                                   yearinfo = yearinfo, 
                                                   questtext = questtext,
                                                   permopt = permopt)
    list_questions = run_lists_questions[0]
    extra_questions = run_lists_questions[1]
    n_perm_opts = run_lists_questions[2]
    data = []
    freq_table_copy = freq_table.copy(deep=True)

    for question in list_questions:
        gpt3_response = call_GPT3(question)
        data.append(gpt3_response) 
    
    res_dat = pd.DataFrame(data, columns = ['question', 
                                            'answer1','answer2','answer3','answer4', 'answer5', 
                                            'score1','score2','score3','score4', 'score5', 
                                            "chosen", "chosen_score", "id", "model", "object"])
    
    
    pindex = [ele for ele in list(freq_table_copy.index) for i in range(n_perm_opts)] 
    res_dat['pindex'] = pindex
    
    #convert to chosen phrases
    charstr='ABCDEFGHIJ'; chars=list(charstr); nums=[i for i in range(0,9)]
    orddict=dict(zip(chars,nums))

    chosen_num = [orddict[i] for i in res_dat['chosen']]
    chosen_opts = [re.sub(r'^[A-Z]\.\s*', '',extra_questions[i][2][chosen_num[i]]) for i in range(0, len(extra_questions), 1)]
    
    pindex_opts_chosen = [(chosen_opts[i],) + list(res_dat['pindex'])[i]  for i in range(0, len(list(res_dat['pindex'])),1)]
    pindex_opts_chosen_unique = list(set(pindex_opts_chosen))
    
    cols_name = list(freq_table_copy.columns)
    cols_name_dup = [x for cols_name in zip(cols_name,cols_name) for x in cols_name]
    gpt_name = ['_GPT3_mean', '_GPT3_sd'] * len(cols_name)
    col_names = [i + j for i, j in zip(cols_name_dup, gpt_name)]
    freq_table_copy[col_names] = ''
    
       
    for q in pindex_opts_chosen_unique:
        
       pos = [True if i == q else False for i in pindex_opts_chosen]
       datsubset = res_dat['chosen_score'].iloc[pos]
       
       try:
           mean_stat = statistics.mean(datsubset)
       except:
           mean_stat = np.nan
  
       try:
           sd_stat = statistics.stdev(datsubset)
       except:
           sd_stat = np.nan
  
       lres = [mean_stat, sd_stat]
       
       var_name = q[0] + '_GPT3_mean'
       pos2 = [True if i == var_name else False for i in freq_table_copy.columns]
       
       freq_table_copy.loc[q[1:len(q)]].iloc[[np.where(pos2)[0][0], np.where(pos2)[0][0]+1]]
       
       
       freq_table_copy.loc[q[1:len(q)],freq_table_copy.columns[[np.where(pos2)[0][0], np.where(pos2)[0][0]+1]]] = lres

    return (res_dat, freq_table_copy)
 
    
def call_GPT3(prompt):
    
    def flatten(A):
        rt = []
        for i in A:
            if isinstance(i,list): rt.extend(flatten(i))
            else: rt.append(i)
        return rt
    
    
    kwargs = {"engine":"text-davinci-003", 
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
    scores["%"] = scores["logprob"].apply(lambda x: np.e**x)
    scores = scores.sort_values(by ='%', ascending=False)
    score_names = [re.sub(r"^\s+|\s+$", "", i).strip()  for i in scores.index]
    chosen = [letter for letter in score_names if letter.isalpha()][0]
    
    chosen_score =  scores[['%']].iloc[[ind for ind, letter in enumerate(score_names) if letter.isalpha()][0]].values.tolist()
    
    res = []; res.append(prompt); res.append(score_names); res.append(list(scores["%"])); res.append(chosen);  res.append(chosen_score);  
    res.append(list(pd.DataFrame([response["id"]])[0])); res.append(list(pd.DataFrame([response["model"]])[0]));
    res.append(list(pd.DataFrame([response["object"]])[0]))
    res = flatten(res)
    
    return(res)