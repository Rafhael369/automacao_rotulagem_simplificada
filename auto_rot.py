import cv2
import queue
import os
import threading
import time
import numpy as np
from darknet.darknet import *
from dotenvs.load_dotenvs import *
from pybboxes import BoundingBox

model = models[0]
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
def camera0(frame, id_camera, model):
    classes, scores, boxes = model.detect(frame, float(os.getenv("CONFIDENCE_THRESHOLD")), float(os.getenv("NMS_THRESHOLD")))
        
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = str(class_names[classid])
            
        centro_box = (int((box[0]+(box[2]/2))), int((box[1]+(box[3])/2)))
        cv2.rectangle(frame[0], box, color, 1)
        cv2.putText(frame[0], label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame[0], centro_box, 5, [0, 0, 0], -1)
    
    caminho_voc = "carro"

    # for (classid, box) in zip(classes, boxes):
    f = open(f"{caminho_voc}.txt", "w")
    f.write(f"{len(boxes)}")
    for box in boxes:
        my_coco_box = [box[0], box[1], box[2], box[3]]
        coco_bbox = BoundingBox.from_coco(*my_coco_box)
        voc_bbox = coco_bbox.to_voc()
        voc_bbox_values = coco_bbox.to_voc(return_values=True)
        voc_txt = f"{int(voc_bbox_values[0])} {int(voc_bbox_values[1])} {int(voc_bbox_values[2])} {int(voc_bbox_values[3])} {classid}"
        f.write(f"\n{voc_txt}")
    f.close()

imagem = cv2.imread("carro.jpg")
camera0(imagem, 0, model)