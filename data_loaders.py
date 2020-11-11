from enum import Enum, unique
import numpy as np
from itertools import chain

import rosbag
from rospy import rostime
import yaml

@unique
class Channels(Enum):
    X_COORD = 0
    Y_COORD = 1
    TIME = 2
    POLARITY = 3


class NumpyEventsAdapter():

    def __init__(self, path, fps):

        self.events = np.load(path)

        dt = 1. / fps
        start_t = self.events[Channels.TIME.value][0]
        stop_t = self.events[Channels.TIME.value][-1]
        frame_ts = np.arange(start_t, stop_t, dt)
        self.frame_ts = np.append(frame_ts, [frame_ts[-1] + dt])
        self.shape = self.events[Channels.Y_COORD.value].max().astype(int) + 1, self.events[Channels.X_COORD.value].max().astype(int) + 1

    def __iter__(self):
        idx_array = np.searchsorted(self.events[Channels.TIME.value], self.frame_ts)
        for i, (idx_begin, idx_end) in enumerate(zip(idx_array[:-1], idx_array[1:])):
            frame_events = self.events[:,idx_begin:idx_end]
            yield i, frame_events, self.shape, self.frame_ts[i], self.frame_ts[i + 1]

    def __len__(self):
        return  len(self.frame_ts) - 1



class RosbagEventsAdapter():

    def __init__(self, path, fps):

        self.bag = rosbag.Bag(path)
        topics = self.bag.get_type_and_topic_info().topics
        self.topics = [topic_name for topic_name, topic_descr in topics.items() if topic_descr.msg_type == 'dvs_msgs/EventArray']

        info_dict = yaml.load(self.bag._get_yaml_info())
        self.duration = info_dict["messages"]

        dt = 1. / fps
        start_t = info_dict["start"]
        stop_t =  info_dict["end"]
        frame_ts = np.arange(start_t, stop_t, dt)
        self.frame_ts = np.append(frame_ts, [frame_ts[-1] + dt])

    def __iter__(self):
        for topic in self.topics[1:]:
            for i, (start_ts, end_ts) in enumerate(zip(self.frame_ts[:-1], self.frame_ts[1:])):
                mathced_msgs = []
                for topic, msg, t in self.bag.read_messages(topics=[topic], start_time=rostime.Time(start_ts), end_time=rostime.Time(end_ts)):
                    iamge_shape = (msg.height, msg.width)
                    mapped_msg = map(lambda event: (event.x, event.y, event.ts.secs * 10 ** 9 + event.ts.nsecs, event.polarity * 2 - 1), msg.events)
                    mathced_msgs = chain(mathced_msgs, mapped_msg)

                if mathced_msgs == []:
                    continue
                frame_events = np.fromiter(mathced_msgs, dtype=[('x', '<f4'), ('y', '<f4'), ('ts', '<f4'), ('polarity', '<f4')])

                # print(" ")
                # print(frame_events.shape, frame_events['polarity'])
                # print(" ")
                yield i, np.array([frame_events['x'], frame_events['y'], frame_events['ts'], frame_events['polarity']]), iamge_shape, start_ts, end_ts


    def __len__(self):
        return self.duration
