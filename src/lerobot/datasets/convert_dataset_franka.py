import os
import numpy as np
import cv2
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_CONFIG = {
    "codebase_version": "v2.1",
    "robot_type": "base_dataset",
    "chunks_size": 1000,
    "fps": 30,
    "video_codec": "mp4v",
    "video_pix_fmt": "yuv420p",
    "camera_configs": {
        "cam": {
            "source": "cam",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"]
        },
        "eih": {
            "source": "eih",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"]
        },
    }
}

def build_features():
    features = {}
    for key, config in DEFAULT_CONFIG["camera_configs"].items():
        features[f"observation.images.{key}"] = {
            "dtype": "video" ,
            "shape": config["shape"],
            "names": config["names"],
            "video_info": {
                "video.fps": DEFAULT_CONFIG["fps"],
                "video.codec": DEFAULT_CONFIG["video_codec"],
                "video.pix_fmt": DEFAULT_CONFIG["video_pix_fmt"],
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
    features.update({
        "observation.state": {
            "dtype": "float64",
            "shape": (8,),
            "names": None
        },
        "action": {
            "dtype": "float64",
            "shape": (8,),  
            "names": None
        },
        "timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": None
        },
        "episode_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "task_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None
        }
    })
    return features

def find_nearest(ts_array, target):
    idx = np.abs(ts_array - target).argmin()
    return idx

def read_episode_data(episode_dir: Path):
    data = {}
    # 统一读取所有 camera_configs
    for cam_key, cam_cfg in DEFAULT_CONFIG["camera_configs"].items():
        cam_dir = episode_dir / cam_key
        images_dir = cam_dir
        image_files = sorted(images_dir.glob("*.png"))
        images = [cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in image_files]
        data[f"observation.images.{cam_key}"] = images

    # angles
    angles_dir = episode_dir / "angles"
    angle_files = sorted(angles_dir.glob("*.npy"))
    angles_values = [np.load(f) for f in angle_files]
    data["observation.state"] = angles_values

    # action
    actions = []
    n = len(angles_values)
    for i in range(n):
        if i == n-1:
            action = angles_values[i]
        else:
            action = angles_values[i+1]
        actions.append(action)
    data["action"] = actions

    duration = 1 / DEFAULT_CONFIG["fps"]
    timestamps = np.arange(0, n * duration, duration)[:n]
    data["timestamp"] = [np.array([t], dtype=np.float64) for t in timestamps]

    data["next.reward"] = [np.array([0.0], dtype=np.float32)] * n
    data["next.done"] = [np.array([False], dtype=bool)] * (n - 1) + [np.array([True], dtype=bool)]
    
    return data

def main(input_dir, output_dir, log_file=None):
    if log_file is not None:
        import sys
        # 同时输出到终端和文件
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        log_file_handle = open(log_file, "w")
        sys.stdout = Tee(sys.stdout, log_file_handle)
        sys.stderr = Tee(sys.stderr, log_file_handle)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    features = build_features()
    ds = LeRobotDataset.create(
        repo_id="my_dataset",
        fps=DEFAULT_CONFIG["fps"],
        features=features,
        root=output_dir,
        robot_type=DEFAULT_CONFIG["robot_type"],
        use_videos=True
    )
    episode_dirs = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]
    for idx, episode_dir in enumerate(episode_dirs):
        if (idx < 100):
            episode_data = read_episode_data(episode_dir)
            n = len(episode_data["timestamp"])
            for i in range(n):
                # 只保留 features 字典中的 key，且只处理 episode_data 中实际存在的 key
                frame = {}
                for k in features:
                    if k in episode_data and k != "timestamp":  # 排除 timestamp
                        frame[k] = episode_data[k][i]
                ds.add_frame(frame, task=str(idx), timestamp=float(episode_data["timestamp"][i][0]))
            ds.save_episode()
            print(f"Episode {idx} done.")
    print("All done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='将数据集转换为LeRobot格式')
    parser.add_argument('--input_dir', default=r'C:\Users\28681\Desktop\robot\3dgs_related\data\base_dataset', help='原始数据集目录')
    parser.add_argument('--output_dir', default=r'C:\Users\28681\Desktop\robot\3dgs_related\data\le_dataset', help='输出LeRobot格式数据集目录')
    parser.add_argument('--log_file', default=None, help='日志输出文件路径（可选，不指定则只输出到终端）')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.log_file)
    
