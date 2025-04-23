import rosbag
import numpy as np
import cv2
import open3d as o3d
from sensor_msgs.msg import Image

# Stereo camera intrinsics (replace with actual values if needed)
focal_length = 278.667
baseline = 0.1

def decode_image(msg: Image):
    """Decode sensor_msgs/Image to numpy array (mono8 or rgb8)"""
    dtype = np.uint8 if msg.encoding in ['mono8', 'rgb8'] else None
    if dtype is None:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    
    img = np.frombuffer(msg.data, dtype=dtype)
    if msg.encoding == 'mono8':
        return img.reshape(msg.height, msg.width)
    elif msg.encoding == 'rgb8':
        return img.reshape((msg.height, msg.width, 3))
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

def get_stereo_images(bag_path, left_topic, right_topic):
    """Get synchronized stereo images from a bag"""
    bag = rosbag.Bag(bag_path)
    left_msgs = []
    right_msgs = []
    
    for topic, msg, t in bag.read_messages(topics=[left_topic, right_topic]):
        if topic == left_topic:
            left_msgs.append(msg)
        elif topic == right_topic:
            right_msgs.append(msg)
    
    bag.close()
    
    min_len = min(len(left_msgs), len(right_msgs))
    left_images = [decode_image(m) for m in left_msgs[:min_len]]
    right_images = [decode_image(m) for m in right_msgs[:min_len]]
    
    return left_images, right_images

def generate_point_cloud(left_img, right_img):
    """Generate 3D point cloud from a stereo image pair"""
    if len(left_img.shape) == 3:
        left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)

    stereo = cv2.StereoSGBM_create(minDisparity=0,
                                   numDisparities=64,
                                   blockSize=9,
                                   P1=8 * 3 * 9 ** 2,
                                   P2=32 * 3 * 9 ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32)

    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan
    depth_map = (focal_length * baseline) / (disparity + 1e-6)

    h, w = left_img.shape
    Q = np.array([[1, 0, 0, -w / 2],
                  [0, -1, 0, h / 2],
                  [0, 0, 0, -focal_length],
                  [0, 0, 1 / baseline, 0]])

    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    mask = np.isfinite(points_3d[:, :, 2])
    points = points_3d[mask]

    return points

def main():
    bag_path = "indoor_forward_7_snapdragon_with_gt.bag"
    left_topic = "/snappy_cam/stereo_l"
    right_topic = "/snappy_cam/stereo_r"

    left_imgs, right_imgs = get_stereo_images(bag_path, left_topic, right_topic)
    print(f"Loaded {len(left_imgs)} stereo pairs.")

    for i, (left, right) in enumerate(zip(left_imgs, right_imgs)):
        print(f"Generating PCD for pair {i}...")
        points = generate_point_cloud(left, right)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        filename = f"cloud_{i}.pcd"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
