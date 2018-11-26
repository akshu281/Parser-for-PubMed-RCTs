from config import *
import json

def extract_json_data(type):
    train_file=open(data_path+type+'.txt')
    lines=train_file.readlines()
    train_set=[]
    for line in lines:
        try:
            if(line.startswith("###")):
                print(line)
                continue;
            else:
                line=line.replace("\"","\'",line.count("\""))
                #print(line)
                #line=line.replace("\n"," ")
                words=line.split("\t")
                #print(words)
                label=words[0]
                text=words[1].split(" ")
                train_set.append({'text_array':text,'text':" ".join(str(word) for word in text),'label':label})
        except:
            print(line)
    with open(type+".json",'w') as tr:
        json.dump(train_set,tr)


extract_json_data('train')
extract_json_data('test')
extract_json_data('dev')
