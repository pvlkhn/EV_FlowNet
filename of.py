from net import Model
import torch
import numpy as np
from functools import partial
from os import path as osp

script_dir = osp.dirname(osp.realpath(__file__))

def _expand(value, shape):
    res = np.zeros(shape, value.dtype)
    s = value.shape
    res[:s[0], :s[1], :s[2]] = value
    return res

class OpticalFlow:
    '''
    The OpticalFrom object computes optical flow

    Parameters
    ----------
        model (str): Name of a file with the model parameters. Default: ./data/model/model.pth
        cuda (bool): Indicator of using GPU. Default: True.
    '''
    def __init__(self,
            model=osp.join(script_dir, 'data/model/model.pth'),
            cuda=True):

        if cuda:
            self._device = torch.device('cuda:0')
            self._back = lambda x: x.cpu().detach().numpy()
        else:
            self._device = torch.device('cpu')
            self._back = lambda x: x.detach().numpy()
        self._net = Model(device=self._device)
        print(model)
        self._net.load_state_dict(torch.load(model, map_location=self._device))
        self._net.to(device=self._device)
        self._net.eval()

    def __call__(self, events, imsize, start, stop, return_all=False):
        ''' Computes optical flow for the input window of events.
        It supports raw events and visualized version of events.

        Parameters
        ----------
            events (tuple): events (x, y, t, p). All entries are iterables. Note: polarities are -1 or 1.
            start (float): a timestamp. A begin of the current window.
            stop (float): a timestamp. An end of the current window.
            return_all (bool): an indicator of returning predictions on every scale. Default: False

        Returns
        -------
            of (np.ndarray): The computed optical flow as 3D tensor with depth 2.
        '''
        flow = self._net(*self._preprocess(events, start, stop), imsize)
        return self._postprocess(flow, return_all)

    def _collate(self, events, start, stop):
        ''' converts tuple of events for each sample to a single Tensor with sample index
        '''
        events = np.hstack([
            np.vstack((
                e,
                np.full_like(e[0], i, dtype=np.float32)))
            for i, e in enumerate(events)
            ])
        start = np.hstack(start)
        stop = np.hstack(stop)
        return (events.T, start, stop)

    def _preprocess(self, events, start, stop):
        return self._collate(events, start, stop)

    def _postprocess(self, flow, return_all):
        back = lambda f: np.transpose(self._back(f), (0, 2, 3, 1))
        if return_all:
            return tuple(map(back, flow))
        return back(flow[-1])
