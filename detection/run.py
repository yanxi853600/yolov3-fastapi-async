import io

from detection.models import *
from detection.utils.utils import *
from detection.utils.datasets import *
from detection.utils.augmentations import *
from detection.utils.transforms import *
from detection.return_json import Bbox, Pred, Images

import os
import sys
import time
import datetime
from datetime import date
import argparse
import shutil
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from werkzeug.utils import secure_filename
import cv2
import logging
import logging.config
# from Inference.errors import Error
import nest_asyncio
#from afs2datasource import DBManager, constant
import threading #parallel thread
from queue import Queue
from typing import List
import json
from fastapi.responses import JSONResponse
from Inference.errors import Error
import asyncio
nest_asyncio.apply()

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
    
    # Need to Setup port, model_name, num_pad, checkpoint_model.
    parser.add_argument("--port", type=str, default="9227",
                        help="port")
    parser.add_argument("--model_name", type=str, default="R0402",
                        help="model_name")
    parser.add_argument("--num_pad", type=int, default="2",
                        help="number of pad")                    
    parser.add_argument("--checkpoint_model", type=str, default="./detection/checkpoints/R0402/R0402_V2_ckpt_245.pth",
                        help="checkpoint_model")
    parser.add_argument("--gpu", type=int, default="0",
                        help="number of pad")   
    parser.add_argument("--conf_thres", type=float, default=0.8,
                        help="The path of the weights to restore.") 
    #
    parser.add_argument("--fraction", type=float, default="0.2",
                        help="number of pad") 
    parser.add_argument("--model_def", type=str, default="./detection/config/yolov3-SMT_AOI.cfg",
                        help="The path of the anchor txt file.")
    parser.add_argument("--class_path", type=str, default="./detection/config/SMT.names",
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--img_size", type=str, default=416,
                        help="The path of the weights to restore.")
    parser.add_argument("--batch_size", type=str, default=1,
                        help="The path of the class names.")
    parser.add_argument("--n_cpu", type=str, default=1,
                        help="The path of the weights to restore.")    
    parser.add_argument("--nms_thres", type=str, default=0.6,
                        help="The path of the weights to restore.")     
                                   
    args = parser.parse_args()
    return args

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = load_classes(args.class_path) 
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

error_logging = Error(args.model_name)
#================= Asynchronously function =========================

#plot_fig
async def plot_fig(loop, imgs, detections, img_detections, uploads_dir, img_filename):
    await asyncio.sleep(0.001)   
    q = Queue()
    all_thread = []
    # 使用 multi-thread
    error_logging.info('--> ' + 'Plot fig started')
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        path = os.path.join(uploads_dir, img_filename)
        print("(%d) Image: '%s'" % (0, path))
        error_logging.info("--> (%d) Saving Image: '%s'" % (0, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        # if detections is not None:# tensor([[ 1.6007e+02,  2.7385e+02,  1.6168e+02,  2.7932e+02,
        detections = rescale_boxes(detections, args.img_size, img.shape[:2]) # (80, 160)
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)

        conf_list = []
        label_list = []
        bbox_list = []

        #Draw 
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            if cls_conf.item() > args.conf_thres:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                error_logging.info('--> '+'\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                
                box_w = x2 - x1
                box_h = y2 - y1
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                conf = cls_conf.item()
                # Add label
                plt.text(
                    x1,
                    y1,
                    s='%s %.2f'%(classes[int(cls_pred)],conf),
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

                conf_list.append(cls_conf.item())
                label_list.append(classes[int(cls_pred)])
                bbox_list.append(x1)
                bbox_list.append(y1)
                bbox_list.append(box_w)
                bbox_list.append(box_h)
    # data insert into queue
    data = []
    data.append(conf_list)
    data.append(label_list)
    data.append(bbox_list)
    #print(data)
    q.put(data)
    all_thread.append(q)

    # 等待全部 Thread 執行完畢
    #for t in all_thread:
        #t.join()
    
    # 使用 q.get() 取出要傳回的值
    result = []
    result.append(q.get())
    # for _ in range(len(all_thread)):
    #     print(q.get())
    #     result.append(q.get())
    
    #print(result)
    error_logging.info('--> ' + 'Plot fig finished')     
    
    return result

#save_fig
async def save_fig(loop, output_path):
    error_logging.info('--> ' + 'Save fig started.')
    await asyncio.sleep(0.1)
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close('all')
    error_logging.info('--> ' + 'Save fig finished') 
    #await asyncio.sleep(0.001)
    print("Save fig finished")
    return True

#copy＆remove image
async def copy_remove(loop, uploads_dir, img_filename, bkp_input_dir):
    await asyncio.sleep(0.1)
    shutil.copy2(os.path.join(uploads_dir, img_filename), os.path.join(bkp_input_dir, img_filename))
    print("copy the image")
    os.remove(os.path.join(uploads_dir, img_filename))
    print("remove the image")
    #await asyncio.sleep(0.001)
    print("copy and remove the image")
    return True

#================= Asynchronously function =========================
def inferenceYoLo(model_load, image, filename):
    error_logging.info('========== request started ==========.')
    try:
        img_size = args.img_size
        conf_thres=args.conf_thres
        nms_thres=args.nms_thres
        today=str(date.today())
        label = 0

        uploads_dir = './output/' +args.model_name +'/'+str(today) + '/uploads/'
        output_dir = './output/' +args.model_name +'/'+str(today) +  '/output/'
        bkp_input_dir = './output/' +args.model_name +'/' +str(today) +  '/bkp_input/'
        log_file = "./detection/logs/" + args.model_name +'/'+ str(today) +'/'
        
        try:
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(bkp_input_dir, exist_ok=True)
            os.makedirs(log_file, exist_ok=True)
        except:
            pass
        
        #log : 要清空handler才可創造log file  >>>  logging.getLogger('').handlers = []
        # logging.basicConfig(level=logging.INFO, filename=log_file + 'SMT.txt', filemode='w', format=' %(asctime)s - %(levelname)s - %(message)s')

        # Request image & save into file
        try:
            file = image
            if not file:
                return {'error': 'Missing file'}, 400
            img_filename = filename
            save_img = Image.open(io.BytesIO(file))
            save_img.save(os.path.join(uploads_dir, img_filename))
            error_logging.info('--> Image request successfully.')
            # error_logging.info('--> Image request successfully.')

        except Exception as e:
            error_logging.warning('  !!! Image request failed : '+ str(e) )

        # Read Image File
        try:
            # image = np.array(os.path.join(uploads_dir, img_filename))
            image = np.array(Image.open(os.path.join(uploads_dir, img_filename)))
            error_logging.info('--> Read image successfully.')
        except Exception as e:
            error_logging.warning('  !!! Read image failed : '+ str(e) )
        

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        # Detect the frame of object & time
        try:
            print("\nPerforming object detection:")
            prev_time = time.time()

            # Configure input
            input_img = transforms.Compose([
                DEFAULT_TRANSFORMS,
                Resize(img_size)])(
                    (image, np.zeros((1, 5))))[0].unsqueeze(0)

            if torch.cuda.is_available():
                input_img = input_img.to("cuda")
                #input_img = Variable(input_img.type(Tensor))

            # Get detections
            with torch.no_grad():
                # detections = load_model(model_name)(input_img)# tensor[1.7, 1.7, 1.4]
                detections = model_load(input_img)
                # [tensor([[1.2806e+02,...000e+00]])]
                detections = non_max_suppression(detections, 2,conf_thres, nms_thres)
                #detections = rescale_boxes(detections[0], img_size, image.shape[:2])

            # Log progress
            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Name %s, Inference Time: %s" % (img_filename, inference_time))
            
            im_p = os.path.join(uploads_dir, img_filename)
            imgs.extend((im_p,))
            img_detections.extend(detections)
            
            error_logging.info('--> Detection object successfully.')

        except Exception as e:
            error_logging.warning('  !!! Detection object failed : '+ str(e) )


        try:
            
            if detections is not None:
                try:
                    loop = asyncio.get_event_loop()
                    # Blocking call which returns when the display_date() coroutine is done
                    result = loop.run_until_complete(plot_fig(loop, imgs, detections, img_detections, uploads_dir, img_filename))
                    #loop.close()

                    # result = plot_fig(imgs, detections, img_detections, uploads_dir, img_filename)
                except Exception as e:
                    error_logging.warning('  !!! Plot fig Failed : ' + str(e))
                    plot_fig(loop, imgs, detections, img_detections, uploads_dir, img_filename).cancel()
                    plot_fig(loop, imgs, detections, img_detections, uploads_dir, img_filename).exception()
                #print(result)
                conf_list, label_list, bbox_list = result[0][0], result[0][1], result[0][2]


                print('conf: ' + str(conf_list))
                print('number of bbox : ' + str(len(conf_list)))
                print('label: '+str(label_list))


                # Every bbox's position coordinate
                new_img_box=[]
                new_img1=[]
                for i in range(len(conf_list)):
                    j = i * 4
                    #print(bbox_list)

                    # print('x1: '+ str(round(bbox_list[j].item(),3)))
                    # print('y1: '+ str(round(bbox_list[j+1].item(),3)))
                    # print('box_w: '+ str(round(bbox_list[j+2].item(),3)))
                    # print('box_h: '+ str(round(bbox_list[j+3].item(),3)))
                    
                    #img_box = Bbox(x1=str(round(bbox_list[j].item(),3)), y2=str(round(bbox_list[j+1].item(),3)), 
                    #                box_w=str(round(bbox_list[j+2].item(),3)), box_h=str(round(bbox_list[j+3].item(),3)))

                    bbox = Bbox(0,0,0,0)
                    bbox.x1=str(round(bbox_list[j].item(),3))
                    bbox.y2=str(round(bbox_list[j+1].item(),3))
                    bbox.box_w=str(round(bbox_list[j+2].item(),3))
                    bbox.box_h=str(round(bbox_list[j+3].item(),3))
                    new_img_box=bbox
                    #print(new_img_box)

                    #img1 = Images(img_name=str(img_filename), pred_cls=str(label_list[i]), conf=float(conf_list[i]), bbox=[img_box])
                    images = Images(0,0,0)
                    images.pred_cls=str(label_list[i])
                    images.conf=float(conf_list[i])
                    images.bbox=new_img_box
                    new_img1.append(images)
                    #print(new_img1.append(images))
                
                pred = Pred(img_name=str(img_filename), predictions=new_img1)
                
                json_data = json.dumps(pred, default=lambda o: o.__dict__, indent=4)
                #print(json_data)
          

                # Every detection's classification
                #for i in range(len(conf_list)):
                    #print(label_list[i])

                label_num = set(label_list)
                #print('label num is: ' + str(len(label_num)))

                try:
                    if len(conf_list) < args.num_pad: # One or less bbox
                        label = "fail"
                        os.makedirs(output_dir+"/fail/", exist_ok=True)

                        output_path = os.path.join(output_dir+"/fail/", f"{img_filename}")
                        try:
                            loop = asyncio.get_event_loop()
                            # Blocking call which returns when the display_date() coroutine is done
                            loop.create_task(save_fig(loop, output_path) )
                            
                        except Exception as e:
                            error_logging.warning('  !!!' + str(e))
                            save_fig(loop, output_path).cancel()
                            save_fig(loop, output_path).exception()
                            pass

                        print("classify to fail file.")
                        print("============ after save image ==============")
                        error_logging.info('--> Detection bbox is one or less.')
                        error_logging.info('--> ' + 'Classify to '+output_path)

                        try:
                            loop = asyncio.get_event_loop()
                            # Blocking call which returns when the display_date() coroutine is done
                            loop.create_task(copy_remove(loop, uploads_dir, img_filename, bkp_input_dir))
                            #loop.close()

                            error_logging.info('--> ' + 'Copy & Remove image is successfully!')
                        
                        except Exception as e:
                            error_logging.warning('  !!!' + str(e))
                            copy_remove(loop, uploads_dir, img_filename, bkp_input_dir).cancel()
                            copy_remove(loop, uploads_dir, img_filename, bkp_input_dir).exception()

                            pass
                        print("============ after copy & remove ==============")

                    else: # two bboxes or more
                        if str(len(label_num)) == '1': # same label

                            if label_list[i] == 'pass': # all pass label
                                label = "pass"
                                os.makedirs(output_dir+"/pass/", exist_ok=True)

                                output_path = os.path.join(output_dir+"/pass/", f"{img_filename}")
                                try:
                                    loop = asyncio.get_event_loop()
                                    # Blocking call which returns when the display_date() coroutine is done
                                    loop.create_task(save_fig(loop, output_path) )
                                    # loop.close()  
                                except Exception as e:
                                    error_logging.warning('  !!!' + str(e))
                                    save_fig(loop, output_path).cancel()
                                    save_fig(loop, output_path).exception() 
                                    pass
                                print("classify to pass file.")
                                print("============ after save image ==============")
                                error_logging.info('--> Detection bboxs are the same label of pass.')
                                error_logging.info('--> Classify to '+output_path)
                                
                            else: # all fail label
                                label = "fail"
                                os.makedirs(output_dir+"/fail/", exist_ok=True)

                                output_path = os.path.join(output_dir+"/fail/", f"{img_filename}")
                                try:
                                    loop = asyncio.get_event_loop()
                                    # Blocking call which returns when the display_date() coroutine is done
                                    loop.create_task(save_fig(loop, output_path) )
                                    # loop.close()  
                                except Exception as e:
                                    error_logging.warning('  !!!' + str(e))
                                    save_fig(loop, output_path).cancel()
                                    save_fig(loop, output_path).exception()
                                    pass
                                print("classify to fail file.")
                                print("============ after save image ==============")
                                error_logging.info('--> Detection bboxs has the label of fail.')
                                error_logging.info('--> ' + img_filename +' is classify to '+output_path)

                                try:
                                    loop = asyncio.get_event_loop()
                                    # Blocking call which returns when the display_date() coroutine is done
                                    loop.create_task(copy_remove(loop, uploads_dir, img_filename, bkp_input_dir))
                                    # loop.close()
                                    error_logging.info('--> ' + 'Copy & Remove image is successfully!')
                                
                                except Exception as e:
                                    error_logging.warning('  !!!' + str(e))
                                    copy_remove(loop, uploads_dir, img_filename, bkp_input_dir).cancel()
                                    copy_remove(loop, uploads_dir, img_filename, bkp_input_dir).exception()
                                    pass
                                print("============ after copy & remove ==============")

                        else: # diff label
                            label = "fail"
                            os.makedirs(output_dir+"/fail/", exist_ok=True)

                            output_path = os.path.join(output_dir+"/fail/", f"{img_filename}")
                            try:
                                loop = asyncio.get_event_loop()
                                # Blocking call which returns when the display_date() coroutine is done
                                loop.create_task(save_fig(loop, output_path) )
                                # loop.close()
                            except Exception as e:
                                error_logging.warning('  !!!' + str(e))
                                save_fig(loop, output_path).cancel()
                                save_fig(loop, output_path).exception()
                                pass
                            print("classify to fail file.")
                            print("============ after save image ==============")
                            error_logging.info('--> Detection bboxs has the label of fail.')
                            error_logging.info('--> ' + img_filename +' is classify to '+output_path)
                            
                            try:
                                loop = asyncio.get_event_loop()
                                # Blocking call which returns when the display_date() coroutine is done
                                loop.create_task(copy_remove(loop, uploads_dir, img_filename, bkp_input_dir))
                                # loop.close()
                                error_logging.info('--> ' + 'Copy & Remove image is successfully!')
                            
                            except Exception as e:
                                error_logging.warning('  !!!' + str(e))
                                copy_remove(loop, uploads_dir, img_filename, bkp_input_dir).cancel()
                                copy_remove(loop, uploads_dir, img_filename, bkp_input_dir).exception()
                                pass
                            print("============ after copy & remove ==============")

                except Exception as e:
                    error_logging.warning('  !!! Caculate the bbox of detection failed : '+ str(e) )

               


            else: # detection=null
                label = "fail"
                os.makedirs(output_dir+"/fail/", exist_ok=True)

                output_path = os.path.join(output_dir+"/fail/", f"{img_filename}")
                try:
                    loop = asyncio.get_event_loop()
                    # Blocking call which returns when the display_date() coroutine is done
                    loop.create_task(save_fig(loop, output_path) )
                    # loop.close()  
                except Exception as e:
                    error_logging.warning('  !!!' + str(e))
                    save_fig(loop, output_path).cancel()
                    save_fig(loop, output_path).exception()
                    pass

                print("Detection is null, classify to fail file.")
                print("============ after save image ==============")
                error_logging.info('--> ' + img_filename +' is classify to '+output_path)

                try:
                    loop = asyncio.get_event_loop()
                    # Blocking call which returns when the display_date() coroutine is done
                    loop.create_task(copy_remove(loop, uploads_dir, img_filename, bkp_input_dir))
                    # loop.close()
                    error_logging.info('--> ' + 'Copy & Remove image is successfully!')
                
                except Exception as e:
                    error_logging.warning('  !!!' + str(e))
                    copy_remove(loop, uploads_dir, img_filename, bkp_input_dir).cancel()
                    copy_remove(loop, uploads_dir, img_filename, bkp_input_dir).exception()
                    pass
                print("============ after copy & remove ==============")

        except Exception as e:
            error_logging.warning('  !!! Detection object failed : '+ str(e) )
        
        error_logging.info('========== request finished ==========.')
        # return json_data
        return label

    except Exception as e:
        error_logging.warning('  !!! Image request is failed : '+ str(e) )


