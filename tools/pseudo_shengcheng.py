import os
from models.networks.deeplabv3 import DEEPLABV3
import torch
from PIL import  Image
import numpy as np
from torchvision import transforms
def load_img_pro(img_path):
    img=Image.open(img_path)
    img=img.resize((512,512))
    img_npy=np.array(img).transpose(2,0,1).astype(np.float32)
    for i in range(img_npy.shape[0]):
        img_npy[i]=(img_npy[i]-img_npy[i].mean())/img_npy[i].std()

    return img_npy
if __name__=='__main__':
    device='cuda'
    source_path= '../dataset/new_coarse_generation'
    target_path='../dataset/MESSIDOR_Base1/pseudo'
    os.makedirs(target_path,exist_ok=True)
    data_list=os.listdir(source_path)
    model=DEEPLABV3()
    model.cuda()
    checkpoint = torch.load('../mobile_checkpoints/checkpoint_200.pth(1).tar')

    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        model.eval()
        for i in data_list:
            content=load_img_pro(os.path.join(source_path,i))
            content=torch.from_numpy(content).cuda().to(dtype=torch.float32).unsqueeze(0)
            output=model(content)
            output_sigmoid=torch.sigmoid(output[0]).cpu().numpy()[0]
            case_seg=np.zeros((512,512))
            case_seg[output_sigmoid[0]>0.5]=255
            case_seg[output_sigmoid[1]>0.5]=128
            case_seg_f=Image.fromarray(case_seg.astype(np.uint8)).resize((800,800),resample=Image.NEAREST)
            case_seg_f.save(os.path.join(target_path,i.replace('.tif','.tif')))





