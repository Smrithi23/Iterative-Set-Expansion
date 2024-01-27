import numpy as np
import json
import math
import sys
from spacy_help_functions import get_entities, create_entity_pairs
from googleapiclient.discovery import build
import spacy
from bs4 import BeautifulSoup
import requests
from spanbert import SpanBERT 
import openai
import re
import time

def gpt(X, sent, open_ai_key, reln):
    openai.api_key = open_ai_key
    models_li= openai.Model.list()
    model = 'text-davinci-003'
    max_tokens = 1000
    temperature = 0.2
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0
    todo = ""
    if reln == "per:schools_attended":
        todo = "If a tuple is a relationship attended school, provide [\"SUBJECT ENTITY\", Attended School, \"OBJECT ENTITY\"]\nExample:  [\"Jeff Bezos\", \"Schools_Attended\", \"Princeton University\"]"
    elif reln == "per:employee_of":
        todo = "If a tuple is a relationship works for, provide [\"SUBJECT ENTITY\", Works For, \"OBJECT ENTITY\"]\nExample: [\"Alec Radford\", \"Work_For\", \"OpenAI\"]"
    elif reln == "per:cities_of_residence":
        todo = "If a tuple is a relationship live in, provide [\"SUBJECT ENTITY\", Live In, \"OBJECT ENTITY\"]\n Example: [\"Mariah Carey", "Live_In", "New York City\"]"
    elif reln == "org:top_members/employees":
        todo = "If a tuple is a relationship top member employee, provide [\"SUBJECT ENTITY\", Top Member Employees, \"OBJECT ENTITY\"]\n Example: [\"Nvidia\", \"Top_Member_Employees\", \"Jensen Huang\"]"
    prompt = str(todo) + str(sent) + "\n"
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    s = response['choices'][0]['text']
    res = re.findall('\[.+\]', s)
    if len(res) < 1:
        return X,0,0
    res = re.sub('\[', '', res[0])
    res = re.sub('\]', '', res)
    res = res.split(',')
    subj = re.sub(r'[^a-zA-Z]', ' ', res[0])
    obj = re.sub(r'[^a-zA-Z]', ' ', res[2])
    subj = subj.strip()
    obj = obj.strip()
    if not subj or not obj:
        return X,0,0
    relation = tuple([subj, obj])
    print("\t\t=== Extracted Relation ===\n")
    text = re.sub(u'\xa0', ' ', str(sent))
    text = re.sub('\t+', ' ', text) 
    text = re.sub('\n+', ' ', text) 
    text = re.sub(' +', ' ', text)
    text = text.replace('\u200b', ' ')
    print("\t\tSentence: {}".format(text))
    print("\t\tSubject: {} ; Object: {} ;".format(relation[0], relation[1]))
    X = set(X)
    old_len = len(X)
    a=0
    X.add(relation)
    if len(X) == old_len:
        print("\t\tDuplicate. Ignoring this.")
    else:
        a=1
        print("\t\tAdding to set of extracted relations")
        print("\t\t==========\n")
    return X,a,1

def spbert(X,candidate_pairs,spanbert,t,reln):
    if len(candidate_pairs) == 0:
        return X,0,0
    a=0
    b=0
    relation_preds = spanbert.predict(candidate_pairs)  # get predictions: list of (relation, confidence) pairs
    for ex, pred in list(zip(candidate_pairs, relation_preds)):
        if pred[0]==reln:
            b+=1
            print("\t\t=== Extracted Relation ===\n")
            print("\t\tInput tokens: ",ex["tokens"])
            print("\t\tOutput Confidence: {} ; Subject: {} ; Object: {} ;".format(pred[1],ex["subj"][0],ex["obj"][0]))
            if pred[1] >= t:
                if (ex["subj"][0],ex["obj"][0]) in X.keys():
                    if pred[1]>X[(ex["subj"][0],ex["obj"][0])]:
                        a+=1
                        print("\t\tAdding to set of extracted relations\n")
                        X[(ex["subj"][0],ex["obj"][0])]=pred[1]
                    else:
                        print("\t\tDuplicate with lower confidence than existing record. Ignoring this.\n")
                else:
                    a+=1
                    print("\t\t\t\tAdding to set of extracted relations\n")
                    X[(ex["subj"][0],ex["obj"][0])]=pred[1]

            else:
                print("\t\tConfidence is lower than threshold confidence. Ignoring this.\n")
            print("\t\t==========")
    return X,a,b
 
def get_reln(r):
    if r==1:
        return "per:schools_attended"
    elif r==2:
        return "per:employee_of"
    elif r==3:
        return "per:cities_of_residence"
    elif r==4:
        return "org:top_members/employees"

def translate_r(r):
    if r==1:
        return "Schools_Attended"
    elif r==2:
        return "Work_for"
    elif r==3:
        return "Live_In"
    elif r==4:
        return "Top_Members_Employees"

def get_subj_obj(r):
    s='PERSON'
    if r==1 or r==2:
        o=['ORGANIZATION']
    elif r==3:
        o=['LOCATION','CITY','STATE_OR_PROVINCE']
    elif r==4:
        s='ORGANIZATION'
        o=['PERSON']
    return s,o

def main():
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    X={}
    cnt=0
    visited=set()
    queried=set()
    model=str(sys.argv[1])
    google_api_key=sys.argv[2]
    search_engine_id=sys.argv[3]
    open_ai_key=sys.argv[4]
    r=int(sys.argv[5])
    t=float(sys.argv[6])
    query=sys.argv[7]
    k=int(sys.argv[8])
    subj_ent,obj_ent=get_subj_obj(r)
    reln=get_reln(r)
    service = build(
        "customsearch", "v1", developerKey=google_api_key
    )
    print("Parameters:")
    print("Client key   = ", google_api_key)
    print("Engine key   = ", search_engine_id)
    print("OpenAI key   = ", open_ai_key)
    print("Method   = ", model[1:])
    print("Relation   = ", translate_r(r))
    print("Threshold   = ", t)
    print("Query   = ", query)
    print("# of Tuples   = ", k)
    print("Loading necessary libraries; This should take a minute or so ...)")
    nlp = spacy.load("en_core_web_lg")
    if model=="-spanbert":
        spanbert = SpanBERT("./pretrained_spanbert")  
    while len(X)<k:
        print("=========== Iteration: {} - Query: {} ===========".format(cnt,query))
        queried.add(query)
        res = (
            service.cse()
            .list(
                q=query,
                cx=search_engine_id,
            )
            .execute()
        )
        urlcount=0
        for rs in res['items']:
            urlcount+=1
            print("URL ( {} / {}): {}".format(urlcount,len(res['items']),rs['link']))
            link=rs['link']
            if link not in visited:
                visited.add(link)
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
                    page = requests.get(link, timeout=30, headers=headers)
                except:
                    return
                if page.status_code != 200:
                    print("\tDid not get 200 response, skipping page")
                    continue
                else:
                    print("\tFetching text from url ...")
                    urlsoup = BeautifulSoup(page.text, "html.parser")
                    text=urlsoup.get_text()
                    # Preprocessing text
                    text = re.sub(u'\xa0', ' ', str(text))
                    text = re.sub('\t+', ' ', text) 
                    text = re.sub('\n+', ' ', text) 
                    text = re.sub(' +', ' ', text)
                    text = text.replace('\u200b', ' ')
                    if len(text)>10000:
                        print("\tTrimming webpage content from {} to 10000 characters".format(len(text)))
                        print("\tWebpage length (num characters): 10000")
                        print("\tAnnotating the webpage using spacy...")
                        doc = nlp(text[:10000])
                    else:
                        print("\tWebpage length (num characters): ", len(text))
                        print("\tAnnotating the webpage using spacy...")
                        doc=nlp(text)
                    sent_num=len(list(doc.sents))
                    sent_cnt=0
                    print("\tExtracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(sent_num))
                    succ=0
                    tot_reln=0
                    sent_extracted=0
                    for sent in doc.sents:
                        # ents=get_entities(sent, entities_of_interest)
                        candidate_pairs=[]
                        sent_entity_pairs = create_entity_pairs(sent,entities_of_interest)
                        for ep in sent_entity_pairs:
                            if ep[1][1]==subj_ent and ep[2][1] in obj_ent:
                                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
                            elif ep[2][1]==subj_ent and ep[1][1] in obj_ent:
                                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject
                        # Classify Relations for all Candidate Entity Pairs using SpanBERT
                        candidate_pairs = [p for p in candidate_pairs if not p["subj"][1] in ["DATE", "LOCATION"]]  # ignore subject entities with date/location type
                        if len(candidate_pairs) > 0 and model=="-spanbert":
                            X,a,b=spbert(X,candidate_pairs,spanbert,t,reln)
                            succ+=a
                            tot_reln+=b
                            if b!=0:
                                sent_extracted+=1
                        elif len(candidate_pairs) > 0 and model == "-gpt3":
                            X,a,b = gpt(X, sent, open_ai_key, reln)
                            time.sleep(1.5)
                            succ+=1
                            tot_reln+=b
                            if b!=0:
                                sent_extracted+=1
                        sent_cnt+=1
                        if sent_cnt%5==0:
                            print("\tProcessed {} / {} sentences\n".format(sent_cnt,sent_num))
                    print("\tExtracted annotations for  {}  out of total  {}  sentences".format(sent_extracted,sent_num))
                    print("\tRelations extracted from this website: {} (Overall: {})".format(succ,tot_reln))
        if len(X)<k:
            if model=="-spanbert":
                print("================== ALL RELATIONS for {} ( {} ) =================".format(reln,len(X)))
                for key, v in sorted(X.items(),key=lambda x: x[1],reverse=True):
                    print("Confidence: {}\t| Subject: {}\t| Object: {}".format(v,key[0],key[1]))
                for key, v in sorted(X.items(),key=lambda x: x[1],reverse=True):
                    q=key[0].lower()+" "+key[1].lower()
                    if q not in queried:
                        query=q
                        break
            elif model=="-gpt3":
                tr=translate_r(r)
                print("================== ALL RELATIONS for {} ( {} ) =================".format(tr,len(X)))
                for rel in X:
                    print("Subject: {}\t| Object: {}".format(rel[0], rel[1]))
                for rel in X:
                    q = rel[0] + " " + rel[1]
                    if q not in queried:
                        query=q
                        break
        if model=="-spanbert":
            print("================== ALL RELATIONS for {} ( {} ) =================".format(reln,len(X)))
            count=0
            for key, v in sorted(X.items(),key=lambda x: x[1],reverse=True):
                print("Confidence: {}\t| Subject: {}\t| Object: {}".format(v,key[0],key[1]))
                count=count+1
                if count>=k:
                    break
            break
        elif model=="-gpt3":
            tr=translate_r(r)
            print("================== ALL RELATIONS for {} ( {} ) =================".format(tr,len(X)))
            count=0
            for rel in X:
                print("Subject: {}\t| Object: {}".format(rel[0], rel[1]))
                count=count+1
                # if count>=k:
                #     break
            break
                                    
if __name__ == "__main__":
    main()
