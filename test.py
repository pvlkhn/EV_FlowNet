import argparse
import numpy as np
import os
from tqdm import tqdm

from imageio import imwrite
from skimage.color import hsv2rgb

from of import OpticalFlow
from data_loaders import NumpyEventsAdapter, RosbagEventsAdapter


def vis_flow(flow):
    mag = np.linalg.norm(flow, axis=2)
    a_mag = np.min(mag)
    b_mag = np.max(mag)

    ang = np.arctan2(flow[...,0], flow[...,1])
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros(list(flow.shape[:2]) + [3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag, 0, 255)
    hsv[:, :, 2] = ((mag - a_mag).astype(np.float32) * (255. / (b_mag - a_mag + 1e-32))).astype(np.uint8)
    flow_rgb = hsv2rgb(hsv)
    return 255 - (flow_rgb * 255).astype(np.uint8)


def vis_events(events, imsize):
    res = np.zeros(imsize, dtype=np.uint8).ravel()
    x, y = map(lambda x: x.astype(int), events[:2])
    i = np.ravel_multi_index([y, x], imsize)
    np.maximum.at(res, i, np.full_like(x, 255, dtype=np.uint8))
    return np.tile(res.reshape(imsize)[..., None], (1, 1, 3))


def collage(flow_rgb, events_rgb):
    flow_rgb = flow_rgb[::-1]

    orig_h, orig_w, c = flow_rgb[0].shape
    h = orig_h + flow_rgb[1].shape[0]
    w = orig_w + events_rgb.shape[1]

    res = np.zeros((h, w, c), dtype=events_rgb.dtype)
    res[:orig_h, :orig_w] = flow_rgb[0]
    res[:orig_h, orig_w:] = events_rgb

    k = 0
    for img in flow_rgb[1:]:
        h, w = img.shape[:2]
        l = k + w
        res[orig_h:orig_h+h, k:l] = img
        k = l
    return res


def apply(arguments):
    events_adapter = RosbagEventsAdapter if arguments.use_bag_file else NumpyEventsAdapter
    events = events_adapter(arguments.input_path, arguments.fps)

    of = OpticalFlow(cuda=arguments.use_cuda)

    os.makedirs(arguments.output_path, exist_ok=True)

    with tqdm(total=len(events)) as progress_bar:
        for i, (progress, frame_events, iamge_shape, frame_start, frame_end) in enumerate(events):
            # events of the current sliding window
            # predicted optical flow. Batch size is equal to 1
            flow = of([frame_events], iamge_shape, [frame_start], [frame_end], return_all=True)
            flow = tuple(map(np.squeeze, flow))
            # visualization
            events_rgb = vis_events(frame_events, iamge_shape)
            flow_rgb = list(map(vis_flow, flow))
            out_path = os.path.join(arguments.output_path, '{:04d}.jpg'.format(i + 1))
            imwrite(out_path, collage(flow_rgb, events_rgb))

            progress_bar.n = progress
            progress_bar.refresh()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply EV_FlowNet")

    cur_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input_path", type=str, default=os.path.join(cur_path, 'data', 'events', 'dvs0.npy'))
    parser.add_argument("--output_path", type=str, default=os.path.join(cur_path, 'output'))
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--use_bag_file", type=bool, default=False)
    parser.add_argument("--fps", type=int, default=120)
    arguments = parser.parse_args()

    apply(arguments)
