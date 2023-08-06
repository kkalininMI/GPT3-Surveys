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
import time
import random
import itertools



openai.api_key = os.environ["OPENAI_API_KEY"]
openai.Model.list()

os.chdir('C:/Users/Kirill/Desktop/GPT-3/polmeth')
#DO NOT RUN:  qdat = pd.read_csv('elites_data_full.csv')

#Year 2020

#Putin
putin_prompts = generate_Prompts(qdat, insertname="Vladimir Putin", numver=3)
comput_results_putin = run_EliteSurvey (putin_prompts, insertname="", insertphrase="In 2020")

#Navalny
navalny_prompts = generate_Prompts(qdat, insertname="Alexey Navalny", numver=3)
comput_results_navalny = run_EliteSurvey (navalny_prompts, insertname="", insertphrase="In 2020")

#state elite
state_prompts = generate_Prompts(qdat, insertname="an average member of the Russian state elite", numver=3)
comput_results_state = run_EliteSurvey (state_prompts, insertname="", insertphrase="In 2020")

#non-state elite
nonstate_prompts = generate_Prompts(qdat, insertname="an average member of the Russian non-state elite", numver=3)
comput_results_nonstate = run_EliteSurvey (nonstate_prompts, insertname="", insertphrase="In 2020")


#Save results
#saveres = {}
#saveres['putin2020'] = [putin_prompts, comput_results_putin]
#saveres['navalny2020'] = [navalny_prompts, comput_results_navalny]
#saveres['state2020'] = [state_prompts, comput_results_state]
#saveres['nonstate2020'] = [nonstate_prompts, comput_results_nonstate]
#pickle.dump(saveres, open("sims2020.pkl", "wb" )) #save object

#load results
#sims2020 = pickle.load(open( "sims2020.pkl", "rb" ) ) #load object
#putin_prompts = sims2020['putin2020'][0];  comput_results_putin = sims2020['putin2020'][1]
#navalny_prompts = sims2020['navalny2020'][0];  comput_results_navalny = sims2020['navalny2020'][1]
#state_prompts = sims2020['state2020'][0];  comput_results_state = sims2020['state2020'][1]
#nonstate_prompts = sims2020['nonstate2020'][0];  comput_results_nonstate = sims2020['nonstate2020'][1]

#sims2022 = pickle.load(open( "sims2022.pkl", "rb" ) ) #load object
#putin_prompts2022 = sims2022['putin2022'][0];  comput_results_putin2022 = sims2022['putin2022'][1]
#navalny_prompts2022 = sims2022['navalny2022'][0];  comput_results_navalny2022 = sims2022['navalny2022'][1]
#state_prompts2022 = sims2022['state2022'][0];  comput_results_state2022 = sims2022['state2022'][1]
#nonstate_prompts2022 = sims2022['nonstate2022'][0];  comput_results_nonstate2022 = sims2022['nonstate2022'][1]


#Year 2022
insertphrase2022 = "In 2022, Russia's invasion of Ukraine intensified the Russo-Ukrainian War, causing mass casualties and destruction. This led to international sanctions and Russia's isolation. Domestically, Russia prioritized a military economy and effectively quashed political opposition to rally support for the war.  In 2022"

#Putin
putin_prompts2022 = generate_Prompts(qdat, insertname="Vladimir Putin", numver=3)
comput_results_putin2022 = run_EliteSurvey (putin_prompts2022, insertname="", insertphrase=insertphrase2022)

#Navalny
navalny_prompts2022 = generate_Prompts(qdat, insertname="Alexey Navalny", numver=3)
comput_results_navalny2022 = run_EliteSurvey (navalny_prompts2022, insertname="", insertphrase=insertphrase2022)

#state elite
state_prompts2022 = generate_Prompts(qdat, insertname="an average member of the Russian state elite", numver=3)
comput_results_state2022 = run_EliteSurvey (state_prompts2022, insertname="", insertphrase=insertphrase2022)

#non-state elite
nonstate_prompts2022 = generate_Prompts(qdat, insertname="an average member of the Russian non-state elite", numver=3)
comput_results_nonstate2022 = run_EliteSurvey (nonstate_prompts2022, insertname="", insertphrase=insertphrase2022)

#Save results
#saveres2022 = {}
#saveres2022['putin2022'] = [putin_prompts2022, comput_results_putin2022]
#saveres2022['navalny2022'] = [navalny_prompts2022, comput_results_navalny2022]
#saveres2022['state2022'] = [state_prompts2022, comput_results_state2022]
#saveres2022['nonstate2022'] = [nonstate_prompts2022, comput_results_nonstate2022]
#pickle.dump(saveres2022, open("sims2022.pkl", "wb" )) #save object

#dict_2020 = {"GPT3_Putin": comput_results_putin[1], 
#             "GPT3_Navalny": comput_results_navalny[1],
#             "GPT3_State elites": comput_results_state[1],
#             "GPT3_Non-state elites": comput_results_nonstate[1],
#             "Survey_State elites": survey_baseline[1],
#             "Survey_Non-state elites": survey_baseline[0]
#             }

#dict_2022 = {"GPT3_Putin": comput_results_putin2022[1], 
#             "GPT3_Navalny": comput_results_navalny2022[1],
#             "GPT3_State elites": comput_results_state2022[1],
#             "GPT3_Non-state elites": comput_results_nonstate2022[1],
#             "Survey_State elites": survey_baseline[1],
#             "Survey_Non-state elites": survey_baseline[0]
#             }


#Obtain proportions from the Survey

dat = pd.read_spss("ICPSR_data2020.sav", convert_categoricals=False)
dat2016 = dat.loc[dat['YEAR']==2016]
dat2020 = dat.loc[dat['YEAR']==2020]

nonstate_elites2020 = dat2020[(dat2020["WORKGROUP"] == 3)|
                          (dat2020["WORKGROUP"] == 4)| 
                          (dat2020["WORKGROUP"] == 1)|
                          (dat2020["WORKGROUP"] == 2)]

state_elites2020 = dat2020[(dat2020["WORKGROUP"] == 9)|
                       (dat2020["WORKGROUP"] == 5)|
                       (dat2020["WORKGROUP"] == 6)]


nonstate_elites2016 = dat2016[(dat2016["WORKGROUP"] == 3)|
                          (dat2016["WORKGROUP"] == 4)| 
                          (dat2016["WORKGROUP"] == 1)|
                          (dat2016["WORKGROUP"] == 2)]

state_elites2016 = dat2016[(dat2016["WORKGROUP"] == 9)|
                       (dat2016["WORKGROUP"] == 5)|
                       (dat2016["WORKGROUP"] == 6)]

survey_baseline = surveyElites(nonstate_elites = nonstate_elites2020, 
                               state_elites = state_elites2020,
                               qdat = qdat)


#Data Analysis
#openres2020 = pickle.load(open("sims2020.pkl", "rb" ))
#openres2022 = pickle.load(open("sims2022.pkl", "rb" ))

dict_2020 = {"GPT3_Putin": openres2020["putin2020"][1][1], 
             "GPT3_Navalny": openres2020["navalny2020"][1][1],
             "GPT3_State elites": openres2020["state2020"][1][1],
             "GPT3_Non-state elites": openres2020["nonstate2020"][1][1],
             "Survey_State elites": survey_baseline[1],
             "Survey_Non-state elites": survey_baseline[0]
             }

dict_2022 = {"GPT3_Putin": openres2022["putin2022"][1][1], 
             "GPT3_Navalny": openres2022["navalny2022"][1][1],
             "GPT3_State elites": openres2022["state2022"][1][1],
             "GPT3_Non-state elites": openres2022["nonstate2022"][1][1],
             "Survey_State elites": survey_baseline[1],
             "Survey_Non-state elites": survey_baseline[0]
             }

#Correlations
keys = list(dict_2020.keys())
key_combinations = itertools.combinations(keys, 2)
results_list=[]
for combination in key_combinations:
    selected_columns = [dict_2020[key] for key in combination]
    result = association_measures(selected_columns[0], 
                                selected_columns[1], 
                                list(combination))
    results_list.append({combination: result[4]})
correlation_mat2020 = construct_correlation_mat(results_list)
#correlation_mat2020.to_csv('correlation_matrix2020.csv', index=False)


keys = list(dict_2022.keys())
key_combinations = itertools.combinations(keys, 2)
results_list2022=[]
for combination in key_combinations:
    selected_columns = [dict_2022[key] for key in combination]
    result = association_measures(selected_columns[0], 
                                selected_columns[1], 
                                list(combination))
    results_list2022.append({combination: result[4]})
correlation_mat2022 = construct_correlation_mat(results_list2022)
#correlation_mat2022.to_csv('correlation_matrix2022.csv', index=False)

#Accuracy
#treatment effect
putin_df = pd.merge(comput_results_putin[1], 
                     comput_results_putin2022[1], left_index=True, right_index=True)
putin_df.to_csv("merged_putin.csv")

navalny_df = pd.merge(comput_results_navalny[1], 
                     comput_results_navalny2022[1], left_index=True, right_index=True)
navalny_df.to_csv("merged_navalny.csv")

#dat1 = comput_results_putin[1]
#dat2 = comput_results_navalny[1]

cramer_res = association_measures(dat1=comput_results_putin[1], 
                                dat2=comput_results_navalny[1], 
                                ndic=["putin", "navalny"])

############################################
#                FUNCTIONS                 #
############################################

##############FUNCTIONS START###############

#Main functions

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def generate_Prompts (qdat, 
                      gptprompt = "Generate semantically similar version of this prompt", 
                      numver=3,
                      insertname="",
                      insertphrase=""):
            
    qdat.set_index("Index", inplace=True, drop=False)
    gdf = pd.DataFrame(columns=["Index", "Variable", "Permutation",	"Questions", 
                                "Option1", "Option2", "Option3", "Option4", "Option5", 
                                "Option6", "Option7", "Option8", "Option9", "Option10"])
    for question in range(1, qdat.shape[0] + 1, 1):
        print(question)    
        newlist = [x for x in qdat.loc[[question]].values.tolist()[0]]
        questionW = newlist[3]
        
        if insertname!="" or insertphrase!="":
            questionW = re.sub(r"\[PLACEHOLDER\]", insertname, questionW)
            questionW = insertphrase + " " + questionW
            
        kwargs = {"engine":"text-davinci-003", 
              "temperature":0.8, 
              "max_tokens":int(len(questionW)/3),
              "top_p":1,
              "frequency_penalty":0, 
              "presence_penalty":0}
        
        prompt = "\""+ questionW + "\"/n" + gptprompt
          
        vers_list = [questionW]
        opts_list = [newlist[4:]] * (numver + 1)
        variable = [newlist[1]] * (numver + 1)
        for ver in range(1, numver + 1, 1):
            secondstowait = random.randint(1, 2)
            time.sleep(secondstowait)
            
            try:
                response = openai.Completion.create(prompt=prompt, **kwargs)
            except:
                time.sleep(60 * 2)
                response = openai.Completion.create(prompt=prompt, **kwargs)
            
            vers_list.append(re.sub(r"\n", "", response["choices"][0]["text"]))

        df = pd.DataFrame([[newlist[1]] * (numver + 1), 
                           [newlist[2]]  * (numver + 1)]).T
        df.columns = ["Variable", "Permutation"]
                
        df_vers = pd.DataFrame(vers_list, columns = ["Questions"])     
        df = df.join(df_vers)         
        
        df_opts = pd.DataFrame(opts_list, columns = ["Option" + str(s) 
                                                for s in list(range(1,len(newlist[4:])+1,1))])
        df = df.join(df_opts)
        
        gdf = pd.concat([gdf, df])
    
    gdf['Index'] = range(1, gdf.shape[0]+1)
        
    return gdf




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
        worded_resp.append(merged_dat[[3]].loc[i].values.tolist()[0][chosenI][3:])
    
    merged_dat['worded_resp'] = worded_resp
    questionN = merged_dat[0].values.tolist()
    questionI = np.unique(questionN).tolist()
    merged_dat['question_wording'] =  [str(i[0]) + ". " + str(i[1]) for i in zip(questionN,  merged_dat[2].values.tolist())]
    ####################
    list_vars = list(np.unique(merged_dat[1]))

    questionsList={}
    questionsListTable={}
    for vars in list_vars:
        merged_dat_s = merged_dat[merged_dat[1]==vars]
        optionI = np.unique(merged_dat_s[['worded_resp']]).tolist()
        optionsList={}
        optionsListTable={}
        for l in optionI:
            
            option_scores = merged_dat_s['chosen_score'][merged_dat_s['worded_resp']==l].values
            mean_scores = statistics.mean(option_scores)
            
            if(len(option_scores)>1):
                sd_scores = statistics.stdev(option_scores)
            else:
                sd_scores = 0  
            
            optionsList[l] = [mean_scores, sd_scores]
            
            l_variant = qdat.columns.values.tolist()[np.where(qdat[qdat['Variable']==vars].iloc[0]==l)[0][0]]
            optionsListTable[l_variant] = [mean_scores, sd_scores]
            
            questionsList[vars]=optionsList
            questionsListTable[vars]=optionsListTable
            tableResult = pd.DataFrame(questionsListTable).T
     
    return(questionsList, tableResult)


def run_GPT3(qdat, insertname, insertphrase):
    run_lists_questions = create_list_questions(qdat=qdat, 
                                                insertname=insertname, 
                                                insertphrase=insertphrase)
    list_questions = run_lists_questions[0]
    extra_questions = run_lists_questions[1]
    data = []

    for question in list_questions:
        
        secondstowait = random.randint(1, 2)
        time.sleep(secondstowait)

        try:
            gpt3_response = call_GPT3(question)
            data.append(gpt3_response)
        except:
            time.sleep(60*2)
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
    list_of_variables = []
    
    for question in range(1, qdat.shape[0]+1, 1):
        print(question)
        newlist = [x for x in qdat.loc[[question]].values.tolist()[0] if pd.isnull(x) == False]
        listopt = ['A. ', 'B. ', 'C. ', 'D. ','E. ','F. ','G. ','H. ','I. ','J. ']
        listoptQ = listopt[0:len(newlist[4:])]
        permopt = newlist[2]
        variables = newlist[1]
        
        if permopt=='yes' or permopt=='Yes' or permopt==True:
            perm_opts = list(permutations(newlist[4:]))
            perm_opts = [j for i in perm_opts for j in i]
        else:
            perm_opts = newlist[4:]
        
        listoptQf = listoptQ * int(len(perm_opts)/len(listoptQ))
        
        perm_opts = [listoptQf[i] + perm_opts[i] for i in range(0, len(perm_opts), 1)]

        questionW = newlist[3]
        
        if insertname!="" or insertphrase!="":
            questionW = re.sub(r"\[PLACEHOLDER\]", insertname, questionW)
            questionW = insertphrase + " " + questionW                

        for opt in range(0, len(perm_opts), len(listoptQ)):
            opts=perm_opts[opt:(opt+len(listoptQ))]
                
            output = str(questionW) + '\n' + '\n'.join(opts) + '\n\nAnswer:'
            list_of_questions.append(output)
            
        for opt in range(0, len(perm_opts), len(listoptQ)):
            opts=perm_opts[opt:(opt+len(listoptQ))]
            list_of_extra.append([question, variables, questionW, opts])
                   
        print("Number of questions generated is " + str(len(list_of_questions)))
    return(list_of_questions, list_of_extra)


def call_GPT3(prompt):
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
    scores["%"] = scores["logprob"].apply(lambda x: 100 * np.e ** x)
    scores = scores.sort_values(by ='%', ascending=False)
    score_names = [re.sub(r"^\s+|\s+$", "", i).strip()  for i in scores.index]
    chosen = [letter for letter in score_names if letter.isalpha()][0]
    
    chosen_score =  scores[['%']].iloc[[ind for ind, letter in enumerate(score_names) if letter.isalpha()][0]].values.tolist()
    
    res = []; res.append(prompt); res.append(score_names); res.append(list(scores["%"])); res.append(chosen);  res.append(chosen_score);  
    res.append(list(pd.DataFrame([response["id"]])[0])); res.append(list(pd.DataFrame([response["model"]])[0]));
    res.append(list(pd.DataFrame([response["object"]])[0]))
    res = flatten(res)
    return(res)

#Data Analysis

def surveyElites(nonstate_elites, state_elites, qdat):
    
    df_nonst = pd.DataFrame(columns=['Option1', 'Option2', 'Option3',
                               'Option4', 'Option5', 'Option6',
                               'Option7', 'Option8', 'Option9', 'Option10'])
    df_stel = pd.DataFrame(columns=['Option1', 'Option2', 'Option3',
                               'Option4', 'Option5', 'Option6',
                               'Option7', 'Option8', 'Option9', 'Option10'])
    for i in qdat['Variable']:
        try:
            nst_e = pd.Series(nonstate_elites[i]).value_counts(normalize=True, sort=False)
            st_e = pd.Series(state_elites[i]).value_counts(normalize=True, sort=False)
           
            nst_e = [[i,0] for i in list(nst_e) if i>0.1]
            st_e = [[i,0] for i in list(st_e) if i>0.1]
            
            df_nonst = df_nonst.append(pd.Series(list(nst_e), index = df_nonst.columns[:len(nst_e)], name = i))
            df_stel = df_stel.append(pd.Series(list(st_e), index= df_stel.columns[:len(st_e)], name = i))
        except:    
            df_nonst = df_nonst.append(pd.Series(np.nan, index = df_nonst.columns[:1], name = i))
            df_stel = df_stel.append(pd.Series(np.nan, index = df_stel.columns[:1], name = i))

            
    return ([df_nonst, df_stel])


def association_measures (dat1, dat2, ndic):
    import numpy as np
    from scipy.stats.contingency import association

    letter_vector = []
    prob_vector = []
    
    list_dat = [dat1, dat2]
    list_let = {}
    list_prob = {}
      
    for d in range(0,len(list_dat)):
        mdat = list_dat[d]
        for i in range(0, mdat.shape[0]):
            letter_vector.append(
                mdat.columns[
                    np.argmax(
                       [j[0] if isinstance(j, list) else j for j in mdat.iloc[i].fillna(0)
                        ])])
            
            prob_vector.append(
                mdat.iloc[i][
                    np.argmax(
                       [j[0] if isinstance(j, list) else j for j in mdat.iloc[i].fillna(0)
                        ])][0])
        
        list_let[ndic[d]] = letter_vector
        letter_vector=[]
        list_prob[ndic[d]] = prob_vector
        prob_vector=[]
        
    rdat = pd.DataFrame(list_let)
    pdat = pd.DataFrame(list_prob)
    
    rdat = rdat.replace(r'Option', "", regex=True)  
    rdat = rdat.astype(int)
     
    confusion_matrix = pd.crosstab(rdat.iloc[:,0], rdat.iloc[:,1])
    accuracy_index = sum(rdat.iloc[:,0] == rdat.iloc[:,1]) / rdat.shape[0]
    sub_pdat = pdat[list(rdat.iloc[:,0]==rdat.iloc[:,1])]
    full_r = np.corrcoef(pdat.iloc[:,0], pdat.iloc[:,1])[0, 1]
    sub_r = np.corrcoef(sub_pdat.iloc[:,0], sub_pdat.iloc[:,1])[0, 1]
    cramers_stats = association(confusion_matrix, method="cramer")
    return([rdat, 
            cramers_stats,
            accuracy_index,
            full_r,
            sub_r])


def construct_correlation_mat(dat):    
    columns, rows = zip(*[list(item.keys())[0] for item in dat])
    df = pd.DataFrame(index=set(columns + rows), columns=set(columns + rows))

    for item in dat:
        key = list(item.keys())[0]
        df.at[key[0], key[1]] = item[key]
        df.at[key[1], key[0]] = item[key]

    df.fillna(1, inplace=True)
    
    gpt3_vars = [var for var in df.index if var.startswith("GPT3_")]
    survey_vars = [var for var in df.index if var.startswith("Survey_")]
    
    sorted_vars = gpt3_vars + survey_vars
    df_sorted = df.reindex(sorted_vars, columns=sorted_vars)
    return (df_sorted)

###############FUNCTIONS END################