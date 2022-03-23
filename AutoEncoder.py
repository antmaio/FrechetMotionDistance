import torch.nn as nn
import torch
import os 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UpSample(nn.Module):
    def __init__(self,feat_in,feat_out,out_shape=None,scale=2):
        super().__init__()
        self.conv = nn.Conv2d(feat_in,feat_out,kernel_size=(3,3),stride=1,padding=1)
        self.out_shape,self.scale = out_shape,scale
        
    
    def forward(self,x):
        return self.conv(
            nn.functional.interpolate(
                x,size=self.out_shape,scale_factor=self.scale,mode='bilinear',align_corners=True))
    
def get_upSamp(feat_in,feat_out, out_shape=None, scale=2, act='relu'):
    
    upSamp = UpSample(feat_in,feat_out,out_shape=out_shape,scale=scale).cuda()
    
    layer = nn.Sequential(upSamp)
    
    if act == 'relu':
        act_f = nn.ReLU(inplace=True).cuda()
        bn = nn.BatchNorm2d(feat_out).cuda()
        layer.add_module('ReLU',act_f)
        layer.add_module('BN',bn)
    elif act == 'sig':
        act_f = nn.Sigmoid()
        layer.add_module('Sigmoid',act_f)
    elif act == 'tanh':
        act_f = nn.Tanh()
        layer.add_module('Tanh', act_f)
    return layer

def add_layer(m,feat_in,feat_out,name,out_shape=None,scale=2,act='relu'):
    upSamp = get_upSamp(feat_in,feat_out,out_shape=out_shape,scale=scale,act=act)
    m.add_module(name,upSamp)
    
def save_onnx(path, name, model):
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.onnx.export(model, 
                     torch.rand((64,3,28,28)).to(device), 
                     os.path.join(path, name), 
                     opset_version=11, 
                     do_constant_folding=True,
                     input_names = ['input'], 
                     output_names = ['output'])