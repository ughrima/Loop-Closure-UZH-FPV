#!/usr/bin/env python

import rospy
import rosbag
from geometry_msgs.msg import PoseStamped
import pandas as pd

def extract_ground_truth_from_bag(bag_file):
    # Open the rosbag file
    bag = rosbag.Bag(bag_file, 'r')

    # List to store ground truth poses
    ground_truth_data = []

    # Loop through the messages in the bag file
    for topic, msg, t in bag.read_messages(topics='/groundtruth/pose'):
        # Extract the timestamp, position, and orientation from PoseStamped
        timestamp = msg.header.stamp.to_sec()
        position = msg.pose.position
        orientation = msg.pose.orientation

        # Store the data
        ground_truth_data.append([timestamp, position.x, position.y, position.z,
                                  orientation.x, orientation.y, orientation.z, orientation.w])

    bag.close()  # Close the rosbag file

    # Save the extracted ground truth data to a CSV file
    df = pd.DataFrame(ground_truth_data, columns=['Timestamp', 'PosX', 'PosY', 'PosZ', 'OriX', 'OriY', 'OriZ', 'OriW'])
    df.to_csv('ground_truth_poses.csv', index=False)
    print("Ground truth data saved to ground_truth_poses.csv")

if __name__ == '__main__':
    # Specify the path to the bag file
    bag_file = '/home/agrima/Desktop/new-approach/indoor_forward_3_snapdragon_with_gt.bag'
    
    # Extract ground truth poses from the bag file
    extract_ground_truth_from_bag(bag_file)
