import os
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import cv2

def load_episode_data(device_root, episode_name, keys):
    ep_dir = os.path.join(device_root, episode_name)
    data = {}
    for k in keys:
        f = os.path.join(ep_dir, f"{k}.npy")
        if os.path.exists(f):
            data[k] = np.load(f)
        else:
            print(f"Warning: {f} does not exist, skipping key '{k}'")
    # 图片目录
    img_dir = os.path.join(ep_dir, "rectify")
    if os.path.exists(img_dir):
        img_files = sorted(os.listdir(img_dir))
        data["rectify_imgs"] = [os.path.join(img_dir, fname) for fname in img_files]
    return data

def find_nearest_idx(arr, val):
    arr = np.asarray(arr)
    idx = np.abs(arr - val).argmin()
    return idx

def get_all_episodes(device_root):
    # 返回所有子文件夹（每个为一个 episode）
    return [d for d in os.listdir(device_root) if os.path.isdir(os.path.join(device_root, d))]

def main():
    src_dir = "C:\\Users\\28681\\Desktop\\robot\\dataset\\demo_main"
    dst_dir = "C:\\Users\\28681\\Desktop\\robot\\dataset\\lerobot_dataset\\sequence_0"
    os.makedirs(dst_dir, exist_ok=True)

    # 设备根目录
    xsense_roots = [os.path.join(src_dir, d) for d in os.listdir(src_dir) if d.startswith("xsense")]
    fish_root = os.path.join(src_dir, "fish_camera")
    vive_root = os.path.join(src_dir, "vive3")

    # 获取所有 episode 名称（假设各设备 episode 文件夹命名一致）
    episode_names = get_all_episodes(vive_root)
    print(f"发现 {len(episode_names)} 个 episode: {episode_names}")

    # 构造 features 字典（以第一个 episode 的数据 shape为准）
    # xsense
    xsense_data_example = []
    for xsense_root in xsense_roots:
        ep_list = get_all_episodes(xsense_root)
        if ep_list:
            data = load_episode_data(xsense_root, ep_list[0], ["timestamp_ms"])
            img_path = data.get("rectify_imgs", [None])[0]
            if img_path:
                img = cv2.imread(img_path)
                xsense_data_example.append(img.shape)
            else:
                xsense_data_example.append(None)
        else:
            xsense_data_example.append(None)
    # fish_camera
    fish_ep_list = get_all_episodes(fish_root)
    fish_img_shape = None
    if fish_ep_list:
        fish_data = load_episode_data(fish_root, fish_ep_list[0], ["timestamp_ms"])
        img_dir = os.path.join(fish_root, fish_ep_list[0], "color")
        if os.path.exists(img_dir):
            img_files = sorted(os.listdir(img_dir))
            if img_files:
                img = cv2.imread(os.path.join(img_dir, img_files[0]))
                fish_img_shape = img.shape
    # vive
    vive_data = load_episode_data(vive_root, episode_names[0], ["xyz", "quat", "timestamp_ms"])

    features = {
        "observation.end_pose": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
    }
    for i, shape in enumerate(xsense_data_example):
        if shape is not None:
            features[f"observation.images.xsense_{i}"] = {
                "dtype": "image",
                "shape": shape,
                "names": ["height", "width", "channels"]
            }
    if fish_img_shape is not None:
        features["observation.images.fish_camera"] = {
            "dtype": "image",
            "shape": fish_img_shape,
            "names": ["height", "width", "channels"]
        }
    features["timestamp"] = {"dtype": "float32", "shape": (1,), "names": None}

    # 创建 LeRobotDataset
    dataset = LeRobotDataset.create(
        repo_id="lerobot/demo_main",
        fps=30,
        features=features,
        root=dst_dir,
        robot_type="custom",
        use_videos=False,
    )

    total_frames = 0
    for ep_idx, ep_name in enumerate(episode_names):
        print(f"处理 episode: {ep_name}")
        # 加载各设备该 episode 的数据
        xsense_datas = [load_episode_data(xsense_root, ep_name, ["timestamp_ms"]) for xsense_root in xsense_roots]
        fish_data = load_episode_data(fish_root, ep_name, ["timestamp_ms"])
        vive_data = load_episode_data(vive_root, ep_name, ["xyz", "quat", "timestamp_ms"])

        # 以最慢设备为基准
        lens = [len(d.get("timestamp_ms", [])) for d in xsense_datas] + \
               [len(fish_data.get("timestamp_ms", [])), len(vive_data.get("timestamp_ms", []))]
        num_frames = min(lens)
        print(f"本 episode 帧数: {num_frames}")
        if num_frames == 0:
            print(f"Warning: episode {ep_name} 没有有效帧，跳过")
            continue

        # 以 fish_camera 时间戳为主线（优先 fish，否则 xsense，否则 vive）
        if fish_data.get("timestamp_ms", None) is not None:
            ref_ts = fish_data["timestamp_ms"]
        elif xsense_datas and xsense_datas[0].get("timestamp_ms", None) is not None:
            ref_ts = xsense_datas[0]["timestamp_ms"]
        elif vive_data.get("timestamp_ms", None) is not None:
            ref_ts = vive_data["timestamp_ms"]
        else:
            print(f"Warning: episode {ep_name} 没有参考时间戳，跳过")
            continue

        # fish_camera图片列表
        fish_imgs = []
        img_dir = os.path.join(fish_root, ep_name, "color")
        if os.path.exists(img_dir):
            img_files = sorted(os.listdir(img_dir))
            fish_imgs = [os.path.join(img_dir, fname) for fname in img_files]

        # 添加帧
        for i in range(num_frames):
            frame = {}
            ts = ref_ts[i]
            #frame["timestamp"] = np.array([ts / 1000.0], dtype=np.float32)
            # 末端位置和姿态（vive）
            idx_vive = find_nearest_idx(vive_data["timestamp_ms"], ts)
            xyz = vive_data["xyz"][idx_vive]
            quat = vive_data["quat"][idx_vive]
            frame["observation.end_pose"] = np.concatenate([xyz, quat]).astype(np.float32)
            # xsense图片
            for j, xsense in enumerate(xsense_datas):
                idx_img = find_nearest_idx(xsense.get("timestamp_ms", []), ts)
                imgs = xsense.get("rectify_imgs", [])
                if imgs and idx_img < len(imgs):
                    img_path = imgs[idx_img]
                    img_bgr = cv2.imread(img_path)
                    frame[f"observation.images.xsense_{j}"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # fish_camera图片
            if fish_imgs:
                idx_fish = find_nearest_idx(fish_data.get("timestamp_ms", []), ts)
                if idx_fish < len(fish_imgs):
                    img_path = fish_imgs[idx_fish]
                    img_bgr = cv2.imread(img_path)
                    frame["observation.images.fish_camera"] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dataset.add_frame(frame, task="main_record")
            total_frames += 1

        dataset.save_episode()
        print(f"已保存 episode {ep_name}")

    print(f"全部 episode 合并完成，总帧数: {total_frames}")
    print(f"已保存为 lerobotdataset 格式，路径: {dst_dir}")

if __name__ == "__main__":
    main()

