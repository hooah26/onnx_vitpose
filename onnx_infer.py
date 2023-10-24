import onnx
import onnxruntime
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.top_down_eval import keypoints_from_heatmaps
from time import time
from configs.ViTPose_base_coco_256x192 import model as model_cfg


# Define file and model paths
IMG_PATH = "/workspace/ViTPose_pytorch/2.png"
ONNX_PATH = 'vitpose.onnx'

# Load the model configuration
model = ViTPose(model_cfg)
device = next(model.parameters()).device
C, H, W = (3, 256, 192)

# Utility function to convert tensor to numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Load the ONNX model and create an inference session
ort_session = onnxruntime.InferenceSession(ONNX_PATH)

# Load and preprocess the input image
img = Image.open(IMG_PATH)
org_w, org_h = img.size
img_tensor = transforms.Compose(
    [transforms.Resize((H, W)),
     transforms.ToTensor()]
)(img).unsqueeze(0).to(device)

# Print information about image size and scale change
print(f">>> Original image size: {org_h} X {org_w} (height X width)")
print(f">>> Resized image size: {H} X {W} (height X width)")
print(f">>> Scale change: {org_h / H}, {org_w / W}")
start_time = time()
# Run inference on the image
tic = time()
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_tensor)}
heatmaps = ort_session.run(None, ort_inputs)[0]
elapsed_time = time() - tic
print(f">>> Output size: {heatmaps.shape} ---> {elapsed_time:.4f} sec. elapsed [{elapsed_time ** -1: .1f} fps]\n")

# Extract keypoints from heatmaps
points, prob = keypoints_from_heatmaps(
    heatmaps=heatmaps,
    center=np.array([[org_w // 2, org_h // 2]]),
    scale=np.array([[org_w, org_h]]),
    unbiased=True, use_udp=True
)
points = np.concatenate([points[:, :, ::-1], prob], axis=2)

max_conf_idx = np.argmax(points[:, 0, 2])
selected_person_points = points[max_conf_idx:max_conf_idx+1]

for pid, point in enumerate(selected_person_points):
    img = np.array(img)[:, :, ::-1] # RGB to BGR for cv2 modules
    img = draw_points_and_skeleton(img.copy(), point, joints_dict()['coco']['skeleton'], person_index=pid,
                                   points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                   points_palette_samples=10, confidence_threshold=0.4)

    end_time = time()
    print('inference time: ', end_time - start_time)
    # save_name = img_path.replace(".jpg", "_result.jpg")
    # cv2.imwrite(save_name, img)
    base_name, ext = os.path.splitext(IMG_PATH)
    save_name = f'{base_name}_result{ext}'
    cv2.imwrite(save_name, img)