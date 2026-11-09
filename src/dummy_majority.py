import csv

counts = {}
with open('isear-train.csv', 'r') as train:
    trainreader = csv.reader(train) 
    trainrows = list(trainreader) 
    for trainrow in trainrows:
        if len(trainrow) < 2:
            continue
        label = trainrow[0]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    majority = max(counts, key=counts.get) # iterate every key in counts, compare by the values

with open('isear-val.csv', 'r') as dev:
    devreader = csv.reader(dev)
    devrows = list(devreader)

with open('isear-val-dummypre-majority.csv','w',newline = '') as out:
    writer = csv.writer(out)
    for devrow in devrows:
        if len(devrow) < 2:
            continue
        writer.writerow([majority,devrow[1]])

