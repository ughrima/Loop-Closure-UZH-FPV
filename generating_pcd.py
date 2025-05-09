import rosbag
import numpy as np
import cv2
import open3d as o3d
from sensor_msgs.msg import Image

# Calibration values from your camchain yaml
# cam0 intrinsics
fx = 278.66723066149086
fy = 278.48991409740296
cx = 319.75221200593535
cy = 241.96858910358173

# Baseline from T_cn_cnm1 (translation x component)
baseline = 0.07961594300469246  # meters

def decode_image(msg: Image):
    """Decode sensor_msgs/Image to numpy array"""
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
    bag = rosbag.Bag(bag_path)
    left_msgs, right_msgs = [], []

    for topic, msg, _ in bag.read_messages(topics=[left_topic, right_topic]):
        if topic == left_topic:
            left_msgs.append(msg)
        elif topic == right_topic:
            right_msgs.append(msg)

    bag.close()

    min_len = min(len(left_msgs), len(right_msgs))
    left_imgs = [decode_image(m) for m in left_msgs[:min_len]]
    right_imgs = [decode_image(m) for m in right_msgs[:min_len]]
    return left_imgs, right_imgs

def generate_point_cloud(left_img, right_img):
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=9,
        P1=8 * 3 * 9 ** 2,
        P2=32 * 3 * 9 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0.0] = np.nan  # Invalid disparities

    h, w = left_gray.shape
    Q = np.array([[1, 0, 0, -cx],
                  [0, -1, 0, cy],
                  [0, 0, 0, -fx],
                  [0, 0, 1 / baseline, 0]])

    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    mask = np.isfinite(points_3d[:, :, 2])

    points = points_3d[mask]

    # Optional: add color
    if len(left_img.shape) == 3:
        colors = left_img[mask]
    else:
        colors = np.tile(left_gray[mask][:, None], (1, 3)) / 255.0  # fake grayscale

    return points, colors

def main():
    bag_path = "/home/agrima/Desktop/new-approach/bags/indoor_forward_7_snapdragon_with_gt.bag"
    left_topic = "/snappy_cam/stereo_l"
    right_topic = "/snappy_cam/stereo_r"

    left_imgs, right_imgs = get_stereo_images(bag_path, left_topic, right_topic)
    print(f"Loaded {len(left_imgs)} stereo pairs.")

    for i, (left, right) in enumerate(zip(left_imgs, right_imgs)):
        print(f"Generating PCD for pair {i}...")
        points, colors = generate_point_cloud(left, right)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        filename = f"cloud_{i}.pcd"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
