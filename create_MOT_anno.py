import os
path = 'data/train/ai2thor/txt/5/'
annoarr = os.listdir(path)
k = 0
for i in annoarr:
    #print (i)
    p = i[20:-4]
    print (p)
    scene = i[:19]
    print (scene)
    anno = open(path + i).read().strip().split("\n")
    for a in anno:
        #print (a)
        b = str(a).split(' ') 
        #for q in b[1]:

        #print (b[1])
        x = float(b[1])*416 - (float (b[3])*416)/2
        y = float(b[2])*416 - (float (b[4])*416)/2
        w = float(b[3])*416
        h = float(b[4])*416
        l = int(b[0])
        ar = [int(p),0,x,y,w,h,l,-1,-1,-1]
        #print (ar)
        with open('data/train/ai2thor/det/' +scene+'.txt','a') as file:
            file.write(str(ar)[1:-1])
            file.write("\n")
            file.close()