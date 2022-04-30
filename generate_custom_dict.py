import argparse
import os
import numpy as np
import pickle
import json
import time
import csv
from datetime import datetime

def save_dataset(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", help="filename of the dataset")
    opts = parser.parse_args()

    file = open(opts.f)
    csv_reader = csv.reader(file, delimiter=',')
    line_count = 0
    
    data = {}
    data['loc']=[]
    data['morning_amt']=[]
    data['afternoon_amt']=[]
    data['final_t']=[]
    data['final_loc']=[]
    data['arrival_t']=[]  
    subarea_dic = {}

    for row in csv_reader:
        if line_count == 0:
            line_count+=1
        else:
            loc = {}
            morning_amt = {}
            afternoon_amt = {}
            final_t = {}
            final_loc = {}
            arrival_t = {}
            gps = row[1].split(')')
            for ind in gps:
                ind = ind.translate({ord(c): None for c in '{:(,}'})
                try:
                    if ind[0] == " ":
                        ind = ind[1:]
                    ret = ind.split(" ")
                    ## normalize coordinates
                    ret[1] = float(ret[1])/100
                    ret[2] = float(ret[2])/200
                    loc[ret[0]] = [ret[1], ret[2]] ## did not search for max or min yet because how would we save it for future reference?
                except:
                    continue
            morning_amount = row[2].split(',')
            for ind in morning_amount:
                ind = ind.translate({ord(c): None for c in '{ }'})
                ret = ind.split(":")
                morning_amt[ret[0]] = ret[1]
                
            afternoon_amount = row[3].split(',')
            for ind in afternoon_amount:
                ind = ind.translate({ord(c): None for c in '{ }'})
                ret = ind.split(":")
                afternoon_amt[ret[0]] = ret[1]
            
            
            # normalize by dividing by 43200 (seconds in 12 hours)
            arrival_time = row[4].split(',')
            for ind in arrival_time:
                ind = ind.translate({ord(c): None for c in '{()})Timestamp'})
                try:
                    if ind[0] == " ":
                        ind = ind[1:]
                    ret = ind.split(": ")
                    ret[1] = ret[1][1:len(ret[1])-1]
                    datetime_obj = datetime.strptime(ret[1], "%Y-%m-%d %H:%M:%S.%f")
                    time = datetime_obj.time()
                    total_seconds = time.hour*3600 + time.minute*60 + time.second
                    arrival_t[ret[0]] = total_seconds / 43200                   
                except:
                    continue 
            
            # normalize by subtracting total seconds by 43200 and dividing by 43200
            final_time = row[5].split(',')
            for ind in final_time:
                ind = ind.translate({ord(c): None for c in '{()})Timestamp'})
                try:
                    if ind[0] == " ":
                        ind = ind[1:]
                    ret = ind.split(": ")
                    ret[1] = ret[1][1:len(ret[1])-1]
                    
                    datetime_obj = datetime.strptime(ret[1], "%Y-%m-%d %H:%M:%S.%f")
                    time = datetime_obj.time()
                    total_seconds = time.hour*3600 + time.minute*60 + time.second
                    final_t[ret[0]] = total_seconds / 43200
                except:
                    continue
            
            final_location = row[6].split(',')
            for ind in final_location:
                ind = ind.translate({ord(c): None for c in '{ }'})
                ret = ind.split(':')
                final_loc[ret[0]] = ret[1]

            dic_final = row[7].split(']')
            for ind in dic_final:
                ind = ind.translate({ord(c): None for c in '{}['})
                try:
                    if ind[0] == ",":
                        ind = ind[2:]
                    x = ind.split(':')
                    y = x[1].split(',')
                    ret = [int(n) for n in y]
                    subarea_dic[x[0]] = ret
                except:
                    continue
            data['loc'].append(loc)
            data['morning_amt'].append(morning_amt)
            data['afternoon_amt'].append(afternoon_amt)
            data['final_t'].append(final_t)
            data['final_loc'].append(final_loc)
            data['arrival_t'].append(arrival_t)       
    

    pickle_name = opts.f[0:len(opts.f)-4] + ".pkl"
    data['subarea_dic'] = subarea_dic   
    save_dataset(data,pickle_name)
                               
                    
                
                
