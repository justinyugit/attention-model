import argparse
import os
import numpy as np
import pickle
import json
import time
import csv
import ciso8601
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
    
    data = []
    loc = []
    morning_amt = []
    afternoon_amt = []
    final_t = []
    final_loc = []
    arrival_t = []


    for row in csv_reader:
        if line_count == 0:
            line_count+=1
        else:
            gps = row[1].split(')')
            for ind in gps:
                ind = ind.translate({ord(c): None for c in '{:(,}'})
                try:
                    if ind[0] == " ":
                        ind = ind[1:]
                    ret = ind.split(" ")
                    ## normalize coordinates
                    ret[1] = str(float(ret[1])/100) ## (x-min)/(max-min)
                    ret[2] = str(float(ret[2])/200) ## (x-min)/(max-min) ## can take result to revert back to original units
                    loc.append(ret)
                except:
                    continue
            
            morning_amount = row[2].split(',')
            for ind in morning_amount:
                ind = ind.translate({ord(c): None for c in '{ }'})
                ret = ind.split(":")
                morning_amt.append(ret)
            
            afternoon_amount = row[3].split(',')
            for ind in afternoon_amount:
                ind = ind.translate({ord(c): None for c in '{ }'})
                ret = ind.split(":")
                afternoon_amt.append(ret)
            print(afternoon_amt)
            final_time = row[4].split(',')
            for ind in final_time:
                ind = ind.translate({ord(c): None for c in '{()})Timestamp'}) ## forget about unix time of all month day year. just look at time of day. and normalize. 12:00 is 0 for both morning and afternoon - morning will be negative. 
                try:
                    if ind[0] == " ":
                        ind = ind[1:]
                    ret = ind.split(": ")
                    ret[1] = ret[1][1:len(ret[1])-1]
                    ts = ciso8601.parse_datetime(ret[1])
                    ret[1] = time.mktime(ts.timetuple())
                    final_t.append(ret)
                except:
                    continue
            final_location = row[5].split(',')
            for ind in final_location:
                ind = ind.translate({ord(c): None for c in '{ }'})
                ret = ind.split(':')
                final_loc.append(ret)
                
            arrival_time = row[6].split(',')
            for ind in arrival_time:
                ind = ind.translate({ord(c): None for c in '{()})Timestamp'})
                try:
                    if ind[0] == " ":
                        ind = ind[1:]
                    ret = ind.split(": ")
                    ret[1] = ret[1][1:len(ret[1])-1]
                    ts = ciso8601.parse_datetime(ret[1])
                    ret[1] = time.mktime(ts.timetuple())
                    arrival_t.append(ret)
                    
                except:
                    continue    
    

    pickle_name = opts.f[0:len(opts.f)-4] + ".pkl"
    data.append(loc)
    data.append(morning_amt)
    data.append(afternoon_amt)
    data.append(final_t)
    data.append(final_loc)
    data.append(arrival_t)    
    save_dataset(data,pickle_name)
    print(data)       
                
               
                    
                
                
