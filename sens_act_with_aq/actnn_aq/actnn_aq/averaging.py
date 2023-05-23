import torch
import numpy as np
from math import ceil
from actnn_aq.conf import config
import actnn_aq.cpp_extension.minimax as ext_minimax

def averaging(input, average_group_size):
    remainder = input.numel() % average_group_size
    if remainder == 0:
        input = input.reshape(-1,average_group_size)
        input = input.mean(dim=1, keepdim=True).view(1,-1)
        return [input, remainder]
    else:
        input_mean = input.view(1,-1)[:,:input.numel()-remainder]
        input_remainder = input.view(1,-1)[:,input.numel()-remainder:]
        
        input_mean = input_mean.view(-1,average_group_size).mean(dim=1, keepdim=True)
        input_remainder = input_remainder.mean(dim=1,keepdim=True)
        
        input_mean = torch.cat([input_mean,input_remainder],dim=0)
        return [input_mean, remainder]
    
def repeating(input, average_group_size, input_shape, remainder):
    input = input.view(-1,1)
    if remainder == 0:
        input = input.repeat(1, average_group_size).view(input_shape)
    else:
        set = np.prod(input_shape)//average_group_size
        input_mean = input[:set,:].repeat(1,average_group_size).view(-1,1)
        input_remainder = input[set:,:].repeat(1,remainder).view(-1,1)
        input = torch.cat([input_mean,input_remainder],dim=0).view(input_shape)
    return input


def minimax_of_averaging(input, H, group_size=config.group_size):
    padding1=padding2=0
    # calculate minimax of averaging 
    if input.numel() % group_size != 0:
        padding1 = group_size - input.numel() % group_size
        input = torch.cat([input.view(-1,1), torch.zeros([padding1,1], dtype=input.dtype, device=input.device)], 0)
        input = input.view(-1,group_size) #[-1,config.group_size]
        mn, mx = ext_minimax.minimax(input)
    else:
        input = input.view(-1,group_size) #[-1,config.group_size]
        mn, mx = ext_minimax.minimax(input)
    # change shape of input: make input to (-1,H,group_size) for applying quantize kernel of actnn
    if mn.numel() % H != 0:
        padding2 = H - mn.numel() % H 
        input = torch.cat([input.view(-1,1),torch.zeros([padding2*group_size,1], dtype=input.dtype, device=input.device)], 0)
        mn = torch.cat([mn, torch.zeros([padding2], dtype=mn.dtype, device=mn.device)])
        mx = torch.cat([mx, torch.zeros([padding2], dtype=mx.dtype, device=mx.device)])
    return input.view(-1,H,group_size), mn.view(-1,H,1), mx.view(-1,H,1), padding1+padding2*group_size


def average_and_pack(input_groups, padding):
    H = input_groups.size(1)
    input_groups = input_groups.view(input_groups.shape[0],-1) #[N,-1]
    if padding != 0:
        input_groups = input_groups[:,:-padding] # remove padding for averaging 
    before_averaging_shape = input_groups.size()
    input_groups, remainder = averaging(input_groups, config.average_group_size) # perform averaging (flatten, [A,1])
    input_groups, q_min, mx, padding2 = minimax_of_averaging(input_groups, H) # prepair quantization by attaining information of averaging
    return input_groups, q_min, mx, remainder, padding2, before_averaging_shape

def repeat_and_unpack(input_groups, before_average_shape,remainder, padding2):
    input_groups = input_groups.view(-1,1) #[A,1]
    if padding2 != 0: # remove padding for repeating
        input_groups = input_groups[:-padding2,:]
    input_groups = repeating(input_groups, config.average_group_size, before_average_shape, remainder) #[N, -1]
    return input_groups
