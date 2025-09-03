from collections import defaultdict
import os
from pathlib import Path
from typing import List, Tuple

from matplotlib import pyplot as plt
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

FONT_PATH = str(Path(__file__) / "fonts/UbuntuMono-R.ttf")

_rng = np.random.RandomState(0)
_palette = ((_rng.random((3 * 255)) * 0.7 + 0.3) * 255).astype(np.uint8).tolist()
_palette = [0, 0, 0] + _palette


def write_video(path, images, fps=5):
    # images: list of numpy arrays
    root_dir = Path(path).parent
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not isinstance(images[0], np.ndarray):
        images_npy = [image.numpy() for image in images]
    else:
        images_npy = images
    media.write_video(path, images_npy, fps=fps)


def annotate_action_on_image(image: np.ndarray, action: dict, scale: float = 80.0) -> np.ndarray:
    """
    Overlay a simple visualization of the action on the RGB frame.
    - world_vector: drawn as a 2D arrow (x,y components) from the image center
    - rot_axangle: displayed as text (axis*angle)
    - gripper & terminate_episode: displayed as text badges

    Args:
        image: HxWx3 uint8
        action: dict with keys 'world_vector' (3,), 'rot_axangle' (3,), 'gripper' (1,), 'terminate_episode' (1,)
        scale: scalar to convert normalized displacement magnitude into pixels for the arrow length
    Returns:
        np.ndarray HxWx3 uint8 with overlay
    """
    img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img)

    # Font setup (fallback to default if custom font missing)
    try:
        font = ImageFont.truetype(FONT_PATH, 16)
    except Exception:
        font = ImageFont.load_default()

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # Draw translation arrow (x right, y down in image coords). We map world x->+x, world y->-y for a rough top-down mapping.
    if action is not None and "world_vector" in action:
        wx, wy, wz = [float(x) for x in action["world_vector"]]
        end_x = int(round(cx + wx * scale))
        end_y = int(round(cy - wy * scale))
        draw.line([(cx, cy), (end_x, end_y)], fill=(0, 255, 0), width=3)
        # Arrow head
        ah = 8
        draw.ellipse([(end_x - ah, end_y - ah), (end_x + ah, end_y + ah)], outline=(0, 255, 0), width=3)

    # Compose info text block
    rot_txt = ""
    if action is not None and "rot_axangle" in action:
        rx, ry, rz = [float(x) for x in action["rot_axangle"]]
        rot_txt = f"rot(ax*ang): [{rx:+.2f}, {ry:+.2f}, {rz:+.2f}]"
    grip_txt = ""
    if action is not None and "gripper" in action:
        grip = float(action["gripper"][0]) if hasattr(action["gripper"], "__len__") else float(action["gripper"])
        grip_txt = f"grip: {grip:+.2f}"
    term_txt = ""
    if action is not None and "terminate_episode" in action:
        term = float(action["terminate_episode"][0]) if hasattr(action["terminate_episode"], "__len__") else float(action["terminate_episode"])
        term_txt = f"term: {int(term > 0.5)}"

    lines = []
    if rot_txt:
        lines.append(rot_txt)
    if grip_txt:
        lines.append(grip_txt)
    if term_txt:
        lines.append(term_txt)

    if lines:
        text = "\n".join(lines)
        # Draw background box for readability
        text_w, text_h = draw.multiline_textsize(text, font=font, spacing=2)
        pad = 6
        box_xy = [(10, 10), (10 + text_w + 2 * pad, 10 + text_h + 2 * pad)]
        draw.rectangle(box_xy, fill=(0, 0, 0))
        draw.multiline_text((10 + pad, 10 + pad), text, fill=(255, 255, 255), font=font, spacing=2)

    return np.asarray(img)


def plot_pred_and_gt_action_trajectory(predicted_actions, gt_actions, stacked_images):
    """
    Plot predicted and ground truth action trajectory
    Args:
        predicted_actions: list of dict with keys as ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']
        gt_actions: list of dict with keys as ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']
        stacked_images: np.array, [H, W * n_images, 3], uint8 (here n_images does not need to be the same as the length of predicted_actions or gt_actions)
    """

    action_name_to_values_over_time = defaultdict(list)
    predicted_action_name_to_values_over_time = defaultdict(list)
    figure_layout = [
        "terminate_episode_0",
        "terminate_episode_1",
        "terminate_episode_2",
        "world_vector_0",
        "world_vector_1",
        "world_vector_2",
        "rotation_delta_0",
        "rotation_delta_1",
        "rotation_delta_2",
        "gripper_closedness_action_0",
    ]
    action_order = [
        "terminate_episode",
        "world_vector",
        "rotation_delta",
        "gripper_closedness_action",
    ]

    for i, action in enumerate(gt_actions):
        for action_name in action_order:
            for action_sub_dimension in range(action[action_name].shape[0]):
                # print(action_name, action_sub_dimension)
                title = f"{action_name}_{action_sub_dimension}"
                action_name_to_values_over_time[title].append(action[action_name][action_sub_dimension])
                predicted_action_name_to_values_over_time[title].append(
                    predicted_actions[i][action_name][action_sub_dimension]
                )

    figure_layout = [["image"] * len(figure_layout), figure_layout]

    plt.rcParams.update({"font.size": 12})

    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    for i, (k, v) in enumerate(action_name_to_values_over_time.items()):

        axs[k].plot(v, label="ground truth")
        axs[k].plot(predicted_action_name_to_values_over_time[k], label="predicted action")
        axs[k].set_title(k)
        axs[k].set_xlabel("Time in one episode")

    axs["image"].imshow(stacked_images)
    axs["image"].set_xlabel("Time in one episode (subsampled)")

    plt.legend()
    plt.show()
