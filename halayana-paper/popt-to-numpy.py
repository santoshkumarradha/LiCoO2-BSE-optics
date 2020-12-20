import numpy as np

print("Reading file\n")
with open('popt.temp', 'r') as fhand:
    lines = [line[:-1] for line in fhand if line.strip() != '']

# index=[i for i in range(len(lines)) if len(list(filter(None,lines[i].split(" "))))==7 ]

# a=[]
# for i in index:
#  a.append(list(map(float,filter(None,lines[1].split(" ")))))
# a=np.array(a)

print("extracting numbers\n")
a = []
for i in lines[1:]:
    a.append(list(map(float, filter(None, i.split(" ")))))
np.save("opt_data", a)
print("Done\n")