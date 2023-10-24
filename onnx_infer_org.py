IMG_PATH = "/workspace/ViTPose_pytorch/1.png"

import onnx
import onnxruntime
from configs.ViTPose_base_coco_256x192 import model as model_cfg

import cv2
import numpy as np
import matplotlib.pyplot as plt

from time import time
from PIL import Image
from torchvision.transforms import transforms
from models.model import ViTPose

from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.top_down_eval import keypoints_from_heatmaps


model = ViTPose(model_cfg)
device = next(model.parameters()).device
C, H, W = (3, 256, 192)
output_onnx = 'vitpose.onnx'
input_names = ["input_0"]
output_names = ["output_0"]
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_session = onnxruntime.InferenceSession(output_onnx)

# Prepare input data
img = Image.open(IMG_PATH)

org_w, org_h = img.size
print(f">>> Original image size: {org_h} X {org_w} (height X width)")
print(f">>> Resized image size: {H} X {W} (height X width)")
print(f">>> Scale change: {org_h / H}, {org_w / W}")
img_tensor = transforms.Compose(
    [transforms.Resize((H, W)),
     transforms.ToTensor()]
)(img).unsqueeze(0).to(device)

# Feed to model
tic = time()
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_tensor)}
heatmaps = ort_session.run(None, ort_inputs)[0]
# heatmaps = vit_pose(img_tensor).detach().cpu().numpy() # N, 17, h/4, w/4
elapsed_time = time() - tic
print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time ** -1: .1f} fps]\n")

# points = heatmap2coords(heatmaps=heatmaps, original_resolution=(org_h, org_w))
points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w // 2, org_h // 2]]),
                                       scale=np.array([[org_w, org_h]]),
                                       unbiased=True, use_udp=True)
points = np.concatenate([points[:, :, ::-1], prob], axis=2)

# Visualization
for pid, point in enumerate(points):
    img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                   points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                   points_palette_samples=10, confidence_threshold=0.4)

    plt.figure(figsize=(5, 10))
    plt.imshow(img)
    plt.title("Result")
    plt.axis('off')
    plt.show()
