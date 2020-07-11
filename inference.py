#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

import argparse
import os
import time
from networks import *
from spec_dataloader import SpectraDataset, SpectraDataLoader

import math
import numpy as np
import sys
import cv2
import pdb
import pickle
import ntpath
import sys
from arpys import dl

#torch.cuda.set_device(0)

def getfiledata(filename):
    data = dl.load_data(filename)
    return vars(data)

def get_spectrum(opt):
    filename = opt.filename
    energy_axis = opt.energy_axis
    momentum_axis = opt.momentum_axis
    remaining_axis = {"x", "y", "z"}
    remaining_axis.remove(energy_axis)
    remaining_axis.remove(momentum_axis)
    for i in remaining_axis:
        remaining_axis = i

    data = getfiledata(os.path.join(filename))
    arr = data['data']
    energy_scale = data[opt.energy_axis+'scale']
    #the below condition checks if the file is atleast in a correct format
    if(data[remaining_axis+'scale'] is not None and not (arr.shape == (len(data['zscale']),len(data['yscale']), len(data['xscale'])))):
        print(filename, ' :- this file contains wrong data for shape')
        print('shape of data array= ' + str(data['data'].shape) + ' and shape of x, y, and z axis= ' + str(len(data['xscale'])) + ' ' + str(len(data['yscale'])) + ' ' + str(len(data['zscale'])))
        sys.exit()
    #dictionary defines the default ordering of the loaded data
    dic = {'x':2, 'y':1, 'z':0}
    arr = np.transpose(arr, [dic[remaining_axis], dic[momentum_axis], dic[energy_axis]])
    spectras = []
    for i in range(arr.shape[0]):
        spectra = arr[i]
            #defining the size of all spectras as (768, 320). Also, in opencv width and height are given in reversed order during resizing
        spectra = cv2.resize(spectra, dsize=(opt.fine_width, opt.fine_height), interpolation=cv2.INTER_CUBIC).astype('float64')
        spectra = transforms.ToTensor()(spectra)
        spectra = (spectra - torch.min(spectra))/(torch.max(spectra)-torch.min(spectra))
        spectra = ((spectra-0.5)/0.5).type('torch.DoubleTensor')
        spectras.append(spectra)
    print('Total spectrums in the given file are:-' + str(arr.shape[0]))
    return spectras, energy_scale

def get_energy(energy_scale, pred):
    val = math.modf((len(energy_scale)-1)*(pred+1)/2)
    energy = energy_scale[int(val[1])] + val[0]*(energy_scale[int(val[1])+1]-energy_scale[int(val[1])]) 
    return energy

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "fermi_detection")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('--filename', default = None)
    parser.add_argument('--energy_axis', default = None)
    parser.add_argument('--momentum_axis', default = None)

    parser.add_argument("--fine_width", type=int, default = 768)
    parser.add_argument("--fine_height", type=int, default = 320)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_store', help='save checkpoint infos')

    opt = parser.parse_args()
    return opt


def main():
    opt = get_opt()
    print(opt)

    if(opt.filename == None):
        print('No filename given to the programme')
        sys.exit()
    if(opt.energy_axis == None or opt.momentum_axis == None):
        print('Either energy axis or momentum axis is missing from the arguments. Trying to extract this information from the file name.')
        filestr = ntpath.basename(opt.filename)
        print(filestr)
        lis = filestr.split('_')
        for i in range(len(lis)):
            if(lis[i].startswith('ef')):
                opt.energy_axis = lis[i+1][1]
                opt.momentum_axis = lis[i+2][1]
                print('Using energy axis as ' + str(opt.energy_axis) + ' and momentum axis as ' + str(opt.momentum_axis))
                break
        if(opt.energy_axis == None or opt.momentum_axis == None):
            print('Failed to parse filename. You should give energy and momentum axis separately')
            sys.exit()

    spectrums, energy_scale = get_spectrum(opt)

    model = Spec_unet()
    model.load_state_dict(torch.load('./checkpoint_store/fermi_detection/step_040000.pth'))
    model.eval()
    model.double()

    fermi_energy = []
    with torch.no_grad():
        for i in range(len(spectrums)):
            pred = model(torch.unsqueeze(spectrums[i], dim = 0))
            energy = get_energy(energy_scale, pred)
            print('Fermi energy predicted by spectrum ' + str(i) + ' = ' + str(energy))
            fermi_energy.append(energy)
    print()
    print('Final predicted value of fermi level = ' + str(sum(fermi_energy)/len(fermi_energy)))


if __name__ == "__main__":
    main()

