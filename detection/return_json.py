from typing import List
import json

#================= Return JSON =========================
class Bbox(object):
    def __init__(self, x1:float, y2:float, box_w:float, box_h:float):
        self.x1 = x1
        self.y2 = y2
        self.box_w = box_w
        self.box_h = box_h

class Images(object):
    def __init__(self, pred_cls:str, conf:float, bbox: List[Bbox]):
        self.pred_cls = pred_cls
        self.conf = conf
        self.bbox = bbox

class Pred(object):
    def __init__(self,img_name: str, predictions: List[Images]):
        self.img_name = img_name
        self.predictions = predictions

#=======================================================================
# img_box1 = Bbox(x1="78.19", y2="78.19", box_w="78.19", box_h="78.19")
# img_box2 = Bbox(x1="23.19", y2="45.19", box_w="58.19", box_h="66.19")

# bbox1 = Images( pred_cls="pass", conf="0.989", bbox=[img_box1])
# bbox2 = Images( pred_cls="pass", conf="0.923", bbox=[img_box2])

# pred = Pred(img_name="1.jpg",predictions=[bbox1, bbox2])

# json_data = json.dumps(pred, default=lambda o: o.__dict__, indent=4)
# print(json_data)

#=======================================================================
# img_box = Bbox(x1=str(round(bbox_list[j].item(),3)), y2=str(round(bbox_list[j+1].item(),3)), 
#                 box_w=str(round(bbox_list[j+2].item(),3)), box_h=str(round(bbox_list[j+3].item(),3)))
# img1 = Image(img_name=img_filename, pred_cls=label_list[i], conf=conf_list[i], bbox=[img_box])
# pred = Pred(predictions=[img1])
# json_data = json.dumps(pred, default=lambda o: o.__dict__, indent=4)
# print(json_data)