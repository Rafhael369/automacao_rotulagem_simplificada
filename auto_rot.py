import cv2
import os
import time
from darknet.darknet import *
from dotenvs.load_dotenvs import *
from pybboxes import BoundingBox

model = models[0]
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
diretorio = "imagens/"

def camera0(frame, id_camera, model, filename):
    classes, scores, boxes = model.detect(frame, float(os.getenv("CONFIDENCE_THRESHOLD")), float(os.getenv("NMS_THRESHOLD")))
        
    f = open(f"{diretorio}{filename}.txt", "a")
    f.write(f"{len(boxes)}")
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = str(class_names[classid])
            
        centro_box = (int((box[0]+(box[2]/2))), int((box[1]+(box[3])/2)))
        cv2.rectangle(frame[0], box, color, 1)
        cv2.putText(frame[0], label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame[0], centro_box, 5, [0, 0, 0], -1)

        
        # for box in boxes:
        my_coco_box = [box[0], box[1], box[2], box[3]]
        coco_bbox = BoundingBox.from_coco(*my_coco_box)
        voc_bbox = coco_bbox.to_voc()
        voc_bbox_values = coco_bbox.to_voc(return_values=True)
        voc_txt = f"{int(voc_bbox_values[0])} {int(voc_bbox_values[1])} {int(voc_bbox_values[2])} {int(voc_bbox_values[3])} {classid}"
        f.write(f"\n{voc_txt}")
    f.close()

# imagem = cv2.imread("carro.jpg")
# camera0(imagem, 0, model)
start_time = time.time()
contador = 0
# lendo as imagens do diretorio
imagens_jpg = [filename for filename in os.listdir(diretorio) if filename.endswith(".jpg")]

for filename in imagens_jpg:
    imagem = cv2.imread(f"{diretorio}{filename}")
    camera0(imagem, 0, model, filename.split(".")[0])

    contador += 1

    # Calcula o tempo decorrido e estimativa do tempo restante
    elapsed_time = time.time() - start_time
    tempo_restante = (elapsed_time / contador) * (len(os.listdir(diretorio)) - contador)
    print(f"Processado {contador} de {len(imagens_jpg)} imagens. Tempo restante estimado: {tempo_restante:.2f} segundos.")