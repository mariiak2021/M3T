import json
  
# Opening JSON file
f = open('data/train/ai2thor/t/result.json')
  
# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list
k =0
for i in data:
    scene = i["filename"][47:-4]
    file2 = scene.split('-', 1)[0]
    #print (file2)
    #print (scene)
    if file2[-1]=="0":

        if len(i["objects"]) == 0:
            anno = [int(scene.split('-', 1)[1]), 0, 0,0,0,0, 1, -1, -1, -1]
            #print (anno)
            with open('/dstore/home/mkhan/sort/data/train/ai2thor/det/'+file2+'.txt', 'a') as file:
                                            file.write(str(anno)[1:-1])
                                            file.write("\n")
                                            file.close()
        else:
            count = 0
            for obj in i["objects"]:
                
                x = obj["relative_coordinates"]["left_x"]
                y = obj["relative_coordinates"]["top_y"]
                w = obj["relative_coordinates"]["width"]
                h = obj["relative_coordinates"]["height"]
                if x < 0:
                    x = 0
                if y<0:
                    y = 0
                if w<0:
                    w = 0
                if h<0:
                    h =0
                if x>416:
                    x = 416
                if y>416:
                    y = 416
                if w>416:
                    w = 416
                if h>416:
                    h = 416
                cl = obj["class_id"]
                if cl not in [1, 23, 24, 46, 47, 33] and obj["confidence"]>0.5:
                    count = count +1
                    anno = [int(scene.split('-', 1)[1]), 0, x,y,w,h, cl, -1, -1, -1]
                #print (anno)
                    with open('/dstore/home/mkhan/sort/data/train/ai2thor/det/'+file2+'.txt', 'a') as file:
                                            file.write(str(anno)[1:-1])
                                            file.write("\n")
                                            file.close()
                if count == 0:
                    anno = [int(scene.split('-', 1)[1]), 0, 0,0,0,0, 1, -1, -1, -1]
                    with open('/dstore/home/mkhan/sort/data/train/ai2thor/det/'+file2+'.txt', 'a') as file:
                                            file.write(str(anno)[1:-1])
                                            file.write("\n")
                                            file.close()
    
# Closing file
f.close()