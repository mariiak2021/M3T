"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from tqdm import tqdm
import torch
from yolo import predict
from torchvision.ops import roi_align
import cv2
import numpy as np
from torch import nn
import os
import torch.nn.functional as F
import os
from sklearn.metrics.pairwise import cosine_similarity  
import numpy as np
#import tkinter
import matplotlib
matplotlib.use('Agg')#('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

#dets dets2
def iou_batch(bb_test, bb_gt, image, image2):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  #print ("bb_gt",bb_gt)#bb_test = bb_test[:-1]
  bb_gt = bb_gt.tolist()
  bb_test = bb_test.tolist()
  for i in bb_gt:
    i[:]=i[:5]
  for i in bb_test:
    i[:]=i[:5]
  #bb_gt = bb_gt[:-1]
  #bb_gt = np.expand_dims(bb_gt, 0)
  ##print ("bb_gt",bb_gt)
  ##print ("bb_test",bb_test)
  #bb_test = np.expand_dims(bb_test, 1)
  boxes_gt = torch.as_tensor(bb_gt).cuda()
  features2 = predict (image2, 'cuda', n_classes = 48, weightfile = './weights/yolo-obj1.pth', height = 416, width = 416)
  ratio = 416/features2.size()[2]
  roi_gt = roi_align(features2, boxes_gt, 3, 1/ratio).cuda()
  roi_gt = F.adaptive_avg_pool2d(roi_gt, (1, 1)).cuda().squeeze()
  #print (roi_gt.shape)   
  boxes_test = torch.as_tensor(bb_test).cuda()
  features = predict (image, 'cuda', n_classes = 48, weightfile = './weights/yolo-obj1.pth', height = 416, width = 416)
  #ratio = 416/features.size()[2]
  #print (boxes_test)
  #if boxes_test.numel() == 0:  # Check if there are no elements in the tensor
  #  boxes_test = torch.tensor([[0., 0., 0., 0., 0.]]).cuda()
  roi_test = roi_align(features, boxes_test, 3, 1/ratio).cuda()
  roi_test = F.adaptive_avg_pool2d(roi_test, (1, 1)).cuda().squeeze()
  ##print (roi_test.shape)    
  if roi_gt.ndim == 1:
    roi_gt = roi_gt[None,:]
    #print (roi_gt.shape) 
  if roi_test.ndim == 1:
    roi_test = roi_test[None,:]
    #print (roi_test.shape)
  #cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
  
  #cos_sim1 = cos1(roi_gt[0],
  #            roi_test[0])
  #cos_sim2 = cos1(roi_gt[0],
  #            roi_test[1])
  #cos_sim3 = cos1(roi_gt[1],
  #            roi_test[0])
  #cos_sim4 = cos1(roi_gt[1],
  #            roi_test[1])
  #print (cos_sim1, cos_sim2, cos_sim3, cos_sim4)
  matrix_1 = roi_test.cpu().detach().numpy()     
  #print (matrix_1)
  
  matrix_2 = roi_gt.cpu().detach().numpy()     
  #print (matrix_2)
  o = cosine_similarity(matrix_2, matrix_1)
  
  #print('\nCosine similarity for the first feature map 13x13: {0}\n'.format(o))
  ##print (o)                         
  return(o)  


class BoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = bbox [:5]
    self.time_since_update = 0
    self.id = BoxTracker.count
    BoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.original_id = bbox[5]

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf=bbox [:5]
    self.original_id = bbox[5]

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    #self.kf = bbox
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    #print ("inside predict append self.kf", self.kf)
    self.history.append(self.kf)
    ##print ("self.histor", self.history[-1][:5])
    return self.history[-1][:5]
  
  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf



def associate_detections_to_trackers(trackers,detections, detections2, iou_threshold, image, image2):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(detections)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections2)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections2, detections, image2, image)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    ##print ("a", a)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        ##print ("stack")
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
      ##print ("linear_assignment")
  else:
    matched_indices = np.empty(shape=(0,2))
    ##print ("no matches")
  #print (matched_indices)
  unmatched_detections = []
  for d, det in enumerate(detections2): 
    #print ("d", d)
    #print ("matched_indices[:,0]", matched_indices[:,1])
    if(d not in matched_indices[:,1]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(detections2):
    if(t not in matched_indices[:,0]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      ##print ("not enough consine")
      unmatched_detections.append(m[1])
      unmatched_trackers.append(m[0])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3, length=12):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.new = []
    self.frame_count = 0
    self.length = length

  def update(self, dets, dets2, image, image2):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    ##print ("frame", self.frame_count)
    # get predicted locations from existing trackers.
    #print ("self.trackers",self.trackers)
    #if self.frame_count == 1:
    #  for i in range(len(dets)):
    #    trk = BoxTracker(dets[i, :])
    #    self.trackers.append(trk)
    trks = np.zeros((len(self.trackers), 6))

    ret = []
    ret2 = []
    unmatched = []
    to_del = []
    #trks = dets2
    ##print ("dets img i", dets)
    ##print ("dets2 img i+1", dets2)
    #a = BoxTracker(dets)
    for t, trk in enumerate(trks):

        pos = self.trackers[t].predict()#[0]
        #print (pos)
        trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], 0]
        #if np.any(np.isnan(pos)):
        #  to_del.append(t)
    #trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    #for t in reversed(to_del):
    #    self.trackers.pop(t)
    #for t, trk in enumerate(trks):
    #    pos = det2
    #    print (pos)
    #    trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4]]
    #    if np.any(np.isnan(pos)):
    #        to_del.append(t)
    #trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    #for t in reversed(to_del):
    #    self.trackers.pop(t)
    #for tr in trks:
    #  tr[0] = tr[0] -408
    #  tr[3] = tr[3] -138

    ##print ("trks", trks)
    if self.frame_count == 1:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(trks, dets, dets2, self.iou_threshold, image, image2)
    else:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(trks, trks, dets2, self.iou_threshold, image, image2)
    #print ("trackers after update with match", self.trackers)
    #print ("matched",matched)
    #print ("unmatched_dets",unmatched_dets)
    #print ("self.trackers",self.trackers)
    # update matched trackers with assigned detections
    ##print ("trackers initial", [a.kf for a in self.trackers])
    #if self.frame_count == 1: new = []
    #new = np.zeros((len(dets2), 5))
    #for t, trk in enumerate(new):
    #    trk[:] = [0, 0, 0, 0, 0]
    ##print ("new", self.new)
    last = []
    for m in matched:
        ##print ("matched", m) 
        if self.frame_count == 1:
          for d in dets: #for d in dets2:
            trk = BoxTracker(d)
            self.trackers.append(trk)
            self.new.append([0,0,0,0,0,0])

        #print ("before", self.trackers[m[1]].kf)
        #print (self.trackers[m[0]])
        ###self.trackers[m[1]]=dets[m[0], :]

        self.trackers[m[0]].update(dets2[m[1], :])

        if self.frame_count == 1:
          #new.append (dets[m[1]])
          self.new [m[0]] = dets[m[0]][:5] 
        else:
          #new.append (trks[m[1]])
          self.new [m[0]] = trks[m[0]][:5]
        if self.frame_count == self.length:
          last.append(np.concatenate((dets2[m[1], :][:5], [self.trackers[m[0]].id + 1], [self.trackers[m[0]].original_id])).reshape(1, -1)) 

        #print ("after", self.trackers[m[1]].kf)
    ##print ("trackers after update with match", [a.kf for a in self.trackers])
    ##print ("new after update with match", [a for a in self.new])
    # create and initialise new trackers for unmatched detections
    ##print ("unmatched", unmatched_dets)
    for i in unmatched_dets:
        
        #else:
        trk = BoxTracker(dets2[i, :])
        #print ("trk", trk)
        self.trackers.append(trk)
        self.new.append(trk.get_state())
    ##print ("trackers after update with unmatched", [a.kf for a in self.trackers])
    ##print ("new after update with unmatched", [a for a in self.new])
    i = len(self.trackers)
    ##print ("i", i)
    for trk, trk_prev in zip(reversed(self.trackers),reversed(self.new)):
        ##print ("one")
        d = trk.get_state()#[0]
        i -= 1
        ##print ("d",d, trk.time_since_update, trk.hit_streak, self.min_hits, self.frame_count, self.min_hits)
        if (trk.time_since_update < 1) and (trk.hits >= self.min_hits): #or self.frame_count <= self.min_hits
        #if (trk.time_since_update <= self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
            #print (np.concatenate((d, [trk.id + 1])))
            #print (np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            ret.append(np.concatenate((d, [trk.id + 1], [trk.original_id])).reshape(1, -1)) 
            ret2.append(np.concatenate((trk_prev, [trk.id + 1], [trk.original_id])).reshape(1, -1)) # +1 as MOT benchmark requires positive
    #for n,r in ipnew:
        # remove dead tracklet
        elif (trk.time_since_update <= self.max_age) and (trk.hits >= self.min_hits): #or self.frame_count <= self.min_hits
        #if (trk.time_since_update <= self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
            #print (np.concatenate((d, [trk.id + 1])))
            #print (np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            #ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1)) 
            ret2.append(np.concatenate((d, [trk.id + 1], [trk.original_id])).reshape(1, -1))
        elif (trk.time_since_update > self.max_age):
            self.trackers.pop(i)
            self.new.pop(i)
        elif (trk.time_since_update > 0):
            unmatched.append(np.concatenate((trk_prev, [trk.id + 1], [trk.original_id])).reshape(1, -1))

    return1 = np.concatenate(ret) if len(ret) > 0 else np.empty((0, 6))
    return3 = np.concatenate(ret2) if len(ret2) > 0 else np.empty((0, 6))
    
    ##print ("return1", return1)
    ##print ("return13", return3)
    #return2 = np.concatenate(unmatched) if len(unmatched) > 0 else np.empty((0, 6))
    return4 = np.concatenate(last) if len(last) > 0 else last
    return return3, return4

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train/m3t')
    parser.add_argument("--phase2", help="Subdirectory in seq_path.", type=str, default='train/ai2thor')
    parser.add_argument("--scene", help="Scene name", type=str, default='FloorPlan25_physics')
    parser.add_argument("--det", help="Detection file name", type=str, default='FloorPlan25_physics.txt')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=1)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.5)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  phase2 = args.phase2
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('outputm3t'):
    os.makedirs('outputm3t')
  pattern = os.path.join(args.seq_path, phase, "*.txt")#args.det##pattern = os.path.join(args.seq_path, phase, 'det', "*.txt")
  print ("number of examples",len(glob.glob(pattern)), pattern)
  pattern_out = os.path.join("outputm3t/", "*.txt")
  ind_final = []
  #for seq_dets_fn in glob.glob(pattern):
    #if "output/"+seq_dets_fn[23:-4]+".txt" not in glob.glob(pattern_out):
  #size = len([])
  pbar2 = tqdm(total=len(glob.glob(pattern)), desc=f"Process 1", leave=False)
  for seq_dets_fn in glob.glob(pattern):
    #print (seq_dets_fn[-5])
    #if seq_dets_fn[-5] == "0":
    if "outputm3t/"+seq_dets_fn.split('/')[-1].split('.')[0]+".txt" in glob.glob(pattern_out):
      print (seq_dets_fn.split('/')[-1].split('.')[0], "is in", " outputm3t")
    if "outputm3t/"+seq_dets_fn.split('/')[-1].split('.')[0]+".txt" not in glob.glob(pattern_out):
      #print (seq_dets_fn)
      #  print (seq_dets_fn[23:-4])
      seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
      mot_tracker = Sort(max_age=args.max_age, 
                        min_hits=args.min_hits,
                        iou_threshold=args.iou_threshold,
                        length = int(seq_dets[:,0].max())) #create instance of the SORT tracker
      print (seq_dets_fn)
      seq = seq_dets_fn.split('/')[-1].split('.')[0]
      print ("seq", seq)
      labelsPath = ("obj1_416.names")
      LABELS = open(labelsPath).read().strip().split("\n")
      with open(os.path.join('outputm3t', '%s.txt'%(seq)),'w') as out_file:
        print("Processing %s."%(seq))
        #print (int(seq_dets[:,0].max()))
        boxes_total = np.empty((0, 11))
        ids = []
        for frame in range(int(seq_dets[:,0].max())):
          #print (frame)
          frame += 1 #detection and frame numbers begin at 1
          total_frames += 1
          if frame ==int(seq_dets[:,0].max()): #frame == int(seq_dets[:,0].max()) or 
            dets2 = seq_dets[seq_dets[:, 0]==int(seq_dets[:,0].min()), 1:7]
            dets2[:, 3:5] += dets2[:, 1:3] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            fn2 = os.path.join(args.seq_path, phase2, 'img', seq + '-%01d.png'%(int(seq_dets[:,0].min())))
            #print (fn2)
            image2 = cv2.imread(fn2)
            if type(image2) == np.ndarray and len(image2.shape) == 3:  # cv2 image
                image2 = torch.from_numpy(image2.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            image2 = image2.cuda()
            image2 = torch.autograd.Variable(image2)
            dets = seq_dets[seq_dets[:, 0]==frame, 1:7]
            dets[:, 3:5] += dets[:, 1:3] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            fn = os.path.join(args.seq_path, phase2, 'img', seq + '-%01d.png'%(frame))
            #print (fn)
            image = cv2.imread(fn)
            if type(image) == np.ndarray and len(image.shape) == 3:  # cv2 image
                image = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            image = image.cuda()
            image = torch.autograd.Variable(image)
          #elif frame ==1: #frame == int(seq_dets[:,0].max()) or 
          #  dets2 = seq_dets[seq_dets[:, 0]==frame, 1:6]
          #  dets2[:, 3:5] += dets2[:, 1:3] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
          #  fn2 = os.path.join(args.seq_path, phase, 'img', 'HousePlan0__train__0-%01d.png'%(frame))
          #  image2 = cv2.imread(fn2)
          #  if type(image2) == np.ndarray and len(image2.shape) == 3:  # cv2 image
          #      image2 = torch.from_numpy(image2.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
          #  image2 = image2.cuda()
          #  image2 = torch.autograd.Variable(image2)
          #  dets = []
          #  image = image2
          else:
            dets = seq_dets[seq_dets[:, 0]==frame, 1:7]
            dets[:, 3:5] += dets[:, 1:3] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            fn = os.path.join(args.seq_path, phase2, 'img', seq + '-%01d.png'%(frame))
            #print ("image", fn, dets)

            image = cv2.imread(fn)
            if type(image) == np.ndarray and len(image.shape) == 3:  # cv2 image
                image = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            image = image.cuda()
            image = torch.autograd.Variable(image)


            dets2 = seq_dets[seq_dets[:, 0]==(frame+1), 1:7]
            dets2[:, 3:5] += dets2[:, 1:3]
            fn2 = os.path.join(args.seq_path, phase2, 'img', seq + '-%01d.png'%(frame+1))
            #print ("image+1",fn2, dets2)
            image2 = cv2.imread(fn2)
            if type(image2) == np.ndarray and len(image2.shape) == 3:  # cv2 image
              image2 = torch.from_numpy(image2.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            image2 = image2.cuda()
            image2 = torch.autograd.Variable(image2)
          if(display):
              fn = os.path.join(args.seq_path, phase2, 'img', seq + '-%01d.png'%(frame))
              im =io.imread(fn)
              ax1.imshow(im)
              plt.title(seq + ' Tracked Targets')
              
          start_time = time.time()
          
          trackers, lastt = mot_tracker.update(dets, dets2, image, image2)
          ##print ()
          ##print (lastt)
          ##print ()
          cycle_time = time.time() - start_time
          total_time += cycle_time
          #print (frame, trackers)
          boxes = np.zeros((len(trackers), 11)) 
          for d,box in zip(trackers, boxes):
            #print('%d,%d,%d,%f,%f,%f,%f,1,-1,-1,-1'%(frame,d[5],d[6],d[1],d[2],d[3]-d[1],d[4]-d[2]),file=out_file)
            
            if int(d[5]) not in ids:
              ids.append (int(d[5]))
            ##print (ids)
            index = ids.index(int(d[5]))
            ##print (int(d[5]),index)
            box[:] = [frame,index,d[6],d[1],d[2],d[3]-d[1],d[4]-d[2],1,-1,-1,-1]
            if(display):
              d = d.astype(np.int32)
              for label in LABELS:
                o = LABELS.index(label)
                if o == d[6]:
                  lab = label + ' id ' + str(index)
                  ##print(lab)
              ax1.add_patch(patches.Rectangle((d[1],d[2]),d[3]-d[1],d[4]-d[2],fill=False,label=lab,lw=3,ec=colours[index%32,:]))
              plt.legend(loc='upper left', fontsize = 'small', ncol = 2)
          #print ("boxes", boxes)
          #for d in tr:
            #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          #  if(display):
          #    d = d.astype(np.int32)
          #    ax1.add_patch(patches.Rectangle((d[1],d[2]),d[3]-d[1],d[4]-d[2],fill=False,lw=0.5,ec=colours[d[5]%32,:]))

          if(display):
            fig.canvas.flush_events()
            
            plt.draw()
            
            ax1.cla()
          if len(boxes) > 0:
            boxes_total = np.concatenate((boxes, boxes_total) , axis=0)
            #print ("boxes total", boxes_total)
        #print ("boxes total", boxes_total.shape, boxes_total)
      if len(lastt) != 0:
        for last in lastt:
          if int(last [5]) not in ids:
            ids.append (int(last [5]))
          
          index =  ids.index(int(last [5]))
          id_last = index
          result = any(box[2] == last[6] and box[0] == 1 and last[1] == box[3] for box in boxes_total)
          if result == True:
            ##print ("last", last)
            for box in boxes_total:
              if box[2] == last[6] and box[0] == 1 and last[1] == box[3]:
                id_box = box[1]
                #print (ids)
                #print (int(last [5]),index)
            for box in boxes_total:
              if box[1] == id_last:
                ##print (box)
                box[1] = id_last
          else:
            with open(os.path.join('outputm3t', seq + '.txt'),'a') as file2:
              print('%d,%d,%d,%6f,%6f,%6f,%6f,1,-1,-1,-1'%(1,id_last,last[6],last[1],last[2],last[3]-last[1],last[4]-last[2]),file=file2)
        for box in boxes_total:
          with open(os.path.join('outputm3t', seq + '.txt'),'a') as file2:
              print('%d,%d,%d,%6f,%6f,%6f,%6f,1,-1,-1,-1'%(box[0],box[1],box[2],box[3],box[4],box[5],box[6]),file=file2)
      else:
        for box in boxes_total:
          with open(os.path.join('outputm3t', seq + '.txt'),'a') as file2:
              print('%d,%d,%d,%6f,%6f,%6f,%6f,1,-1,-1,-1'%(box[0],box[1],box[2],box[3],box[4],box[5],box[6]),file=file2)
      pbar2.update(1)
  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
  pbar2.close()
  
