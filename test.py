# import rospy
# import rosbag
# import os
# import sys
# from sensor_msgs.msg import JointState

# USER_NAME = 0
# dir = os.path.dirname(os.path.abspath(__file__))
# bagdir = os.path.join(dir,"bags/user{}/{}.bag".format(USER_NAME,USER_NAME))
# print(bagdir)
# bag = rosbag.Bag(bagdir,'r')
# msgs = []

# for topic, msg,t in bag.read_messages(topics=['/my_gen3_lite/joint_states']):
#     temp = JointState()
#     temp.header = msg.header
#     temp.position = msg.position
#     temp.velocity = msg.velocity
#     temp.name = msg.name
#     temp.effort = msg.effort
#     msgs.append(temp)

# print(msgs)

from BC.replay_buffer import SimpleReplayBuffer
from BC.utils.bc_utils import get_all_bag_files,whole_bag_to_messages, read_one_trajectory_to_each_buffer

ls = get_all_bag_files()
msgs = whole_bag_to_messages(ls)
buffers = []
for i in range(10):
    # print(i)
    buf =   SimpleReplayBuffer([6],[6],100)
    buffers.append(buf)
print(buffers[0])
read_one_trajectory_to_each_buffer(10,buffers, msgs)
print(buffers[0].random_sample(10))


# import numpy as np

# size = 6
# if isinstance(size,int):
#     size = [size]
# buffer = np.zeros((10,*size))
# print(buffer)

import torch

# reward = torch.distributions.Normal(torch.tensor([1,1]), torch.tensor([2,3]))
# logprob=reward.log_prob(torch.tensor([0,0]))
# print(logprob.mean())
# print(len(torch.tensor([1,2,3,45,6])))