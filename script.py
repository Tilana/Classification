still_lost = []
c=0
c_notfound = 0
not_found = []
for name in lost:
    st = name.split(';')[0]
    missing_data = features[features['Filename'].str.contains(st)]
    if len(missing_data)==1:
        print 'Match Case'
        c= c+1
        print name
        print missing_data['Filename']
        #data = data.append(missing_data)
    elif len(missing_data)==0:
        c_notfound = c_notfound + 1
        not_found.append(name)
        print 'not found'
        print name
#        print missing_data
    else:
        still_lost.append(name)
        #print 'Too many'
        #print name
        #print missing_data['Filename']

for elem in lost:
    elem = elem.split(';')[0]

