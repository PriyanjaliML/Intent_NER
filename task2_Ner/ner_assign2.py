
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
import json
import re
import spacy

# load english language model
nlp = spacy.load('en_core_web_sm')
json_file = open("isda_json/isda_data.json") #import finanacial textual data


data = json.load(json_file)      #load the json data

rounding = ["nearest","down"]    #add the keywords
currency = ["EUR","USD","GBP"]    
train = []                       #initialize train data
count_round = 0
for i in data:                   
    sentence = i['text']
    
    sen = sentence.split()
    #print(sen)
    count = 0
    
    dict_entities = dict()            #initialize the dictionary for entities
    list_entities = []           
    enter_loop = 0                    #occurence of finding the currency or amount
    list_amount = []                   
    count_round = 0                   #occurence of finding the rounding 
    count_curr = 0
    for each in sen:                  #create ner data from textual data
        
        for word in currency:
            
            if(word == each):
                pattern = word
                
                l = [(pattern, m.start(), m.end())for m in re.finditer(pattern, sentence)] #
                count_curr = count_curr + 1
                if count_curr==1:
                    info_currency = tuple((l[0][1],l[0][2],"currency"));list_entities.append(info_currency)
                    
                word_amount = sen[count+1]
               
                amount_start = sentence.find(word_amount)
                amount_end = amount_start + len(word_amount)-1
                if(enter_loop ==0):
                    info_amount = tuple((amount_start,amount_end,"amount"));list_entities.append(info_amount)
                enter_loop = enter_loop +1    
               
        for word in rounding:
            
            if(word == each):
                word_start = sentence.find(word)
                word_end = word_start + len(word)-1
                info_round = tuple((word_start,word_end,word))
                if(count_round == 0):
                    list_entities.append(info_round)
                count_round = count_round + 1
            
        count = count + 1
    
   
    dict_entities["entities"] = list_entities
    
    create_each_data = tuple((sentence,dict_entities))
    #print(create_each_data)
    train.append(create_each_data)

#train ner 

ner = nlp.get_pipe("ner")
for _,annotations in train:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])
        
disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
print(disable_pipes)       


from spacy.training.example import Example
import random
from spacy.util import minibatch, compounding
from pathlib import Path

with nlp.disable_pipes(*disable_pipes):
    optimizer = nlp.resume_training()

    for iteration in range(100):

        random.shuffle(train)
        losses = {}

        batches = minibatch(train, size=compounding(1.0, 4.0, 1.001))
        print(batches)
        
        
        for batch in batches:
            for text, annotations in batch:
                # create Example
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses, drop=0.1,sgd=optimizer)

        print("Losses", losses)
        

#evaluating the trained ner on textual data
for text, _ in train: 
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])    
#testing on new sentence
text = "I have to deliever the amount of nearest to EUR 10,000"
doc = nlp(text)

for ent in doc.ents:
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    
    
