import csv
import random

labels = ['sadness', 'guilt', 'joy', 'anger', 'shame', 'disgust', 'fear'] 
# I tried to extract tags, but the file format wasn't completely clean. Then I turned to hardcode.

with open('isear-val.csv', 'r') as file:
    reader = csv.reader(file) # read the file and split the col by comma
    rows = list(reader)

with open('isear-val-dummyprediction.csv','w', newline = '') as out:
    writer = csv.writer(out)
    for row in rows:
        if len(row) < 2: # avoid traceback caused by nonstandard data
            print(row)
            continue
        pred = random.choice(labels)
        writer.writerow([pred, row[1]])

