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

from BC.utils.bc_utils import get_all_bag_files,whole_bag_to_messages

ls = get_all_bag_files()
msgs = whole_bag_to_messages(ls)
print(msgs[1])

