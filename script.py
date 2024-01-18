import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

from typing import List, Dict
from dataclasses import dataclass, field
from os import listdir
from os.path import join, isfile, basename, isdir

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Roblox - Take-Home Assignment")
parser.add_argument("path", type=str, help="Path to an image file or a directory of images")
parser.add_argument("--metric", action="store_true", default=False, help="Calculate and print metrics")

@dataclass
class ModelSettings:
    """
    Data class representing model configuration settings.
    
    Attributes:
        trained_model (str): File path to the trained model.
        cfg (Dict): Configuration dictionary for the backbone network.
        confidence_threshold (float): Threshold for confidence score.
        top_k (int): Number of top bounding boxes to keep before NMS.
        nms_threshold (float): IoU threshold for NMS.
    """
    trained_model: str = "./weights/Resnet50_Final.pth"
    cfg: Dict = field(default_factory=lambda: cfg_re50)
    confidence_threshold: float = 0.5
    top_k: int = 5000
    nms_threshold: float = 0.5


def check_keys(model: RetinaFace, pretrained_state_dict: Dict, print_result: bool=False) -> bool:
    """
    Checks if the keys from the model and the pretrained_state_dict match.
    
    Args:
        model (RetinaFace): The model to check.
        pretrained_state_dict (Dict): The state dict from the pretrained model.
        print_result (bool): Flag to print the result.

    Returns:
        bool: True if keys match, False otherwise.
    """
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    
    if print_result:
        print("Missing keys:{}".format(len(missing_keys)))
        print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
        print("Used keys:{}".format(len(used_pretrained_keys)))
    
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    
    return True


def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    """
    Removes a specified prefix from the keys in the state_dict.

    Args:
        state_dict (Dict): The state dictionary to modify.
        prefix (str): The prefix to remove.

    Returns:
        Dict: Modified state dictionary without the prefix.
    """
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model: RetinaFace, pretrained_path: str, load_to_cpu: bool=True) -> RetinaFace:
    """
    Loads a pretrained model from a specified path into the provided RetinaFace model.

    Args:
        model (RetinaFace): The RetinaFace model to load weights into.
        pretrained_path (str): Path to the pretrained model.
        load_to_cpu (bool): Flag to load the model onto the CPU.

    Returns:
        RetinaFace: The model with loaded weights.
    """
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    
    # Old style model is stored with all names of parameters sharing common prefix "module."
    # Need to remove them before proceeding
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    
    check_keys(model, pretrained_dict)
    
    model.load_state_dict(pretrained_dict, strict=False)
    return model
    

def get_file_paths(path: str) -> List[str]:
    """
    Retrieves a list of file paths from the specified directory or file.

    Args:
        path (str): Path to the directory or file.

    Returns:
        List[str]: List of file paths.
    """
    if isfile(path):
        return [path]
    elif isdir(path):
        return [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    else:
        print(f"[Warning] {path} is not a file or directory")
        return []


def setup_RetinaFace_model_inference(model_settings: ModelSettings, load_to_cpu: bool=True) -> RetinaFace:
    """
    Sets up the RetinaFace model for inference with specified model settings.

    Args:
        model_settings (ModelSettings): Settings for the model.
        load_to_cpu (bool): Flag to setup the model on the CPU.

    Returns:
        RetinaFace: The configured RetinaFace model ready for inference.
    """
    net = RetinaFace(cfg=model_settings.cfg, phase = "test")
    net = load_model(net, model_settings.trained_model, load_to_cpu=True)
    net.eval()    
    cudnn.benchmark = True    
    device = torch.device("cpu") if load_to_cpu else torch.cuda.current_device()
    net = net.to(device)
    
    return net
    

def run_RetinaFace_inference(net: RetinaFace, model_settings: ModelSettings, img_raw: np.ndarray) -> np.ndarray:
    """
    Runs the RetinaFace model inference on an image.

    Args:
        net (RetinaFace): The RetinaFace model.
        model_settings (ModelSettings): The settings for the model.
        img_raw (np.ndarray): The raw image array.

    Returns:
        np.ndarray: The detected bounding boxes and landmarks.
    """
    # device for low-level processing
    device = torch.device("cpu")
    
    # image preprocessing
    img = np.float32(img_raw)                
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    
    # inference forward pass
    loc, conf, landms = net(img)  
    
    # decode inference results
    priorbox = PriorBox(model_settings.cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, model_settings.cfg["variance"])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, model_settings.cfg["variance"])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # threshold on confidence score
    inds = np.where(scores > model_settings.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]                        

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, model_settings.nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # combine bbox and landmarks
    dets = np.concatenate((dets, landms), axis=1)   
    
    return dets


class MetricCollector:
    """
    Class for collecting and calculating metrics based on detection results.
    """
    def __init__(self):
        # num of faces in all detected images
        self.all_nfaces_gt = []  # ground truth
        self.all_nfaces_pr = []  # prediction
        
    def collect(self, nfaces_gt: int, nfaces_pr: int):
        """
        Collects ground truth and predicted number of faces for metric calculation.

        Args:
            nfaces_gt (int): Number of ground truth faces.
            nfaces_pr (int): Number of predicted faces.
        """
        self.all_nfaces_gt.append(nfaces_gt)
        self.all_nfaces_pr.append(nfaces_pr)
    
    def calculate_and_print(self):
        """
        Calculates and prints the evaluation metrics.
        """
        if not len(self.all_nfaces_gt) or len(self.all_nfaces_gt) != len(self.all_nfaces_pr):
            print(f"[Warning] unable to calculate metrics")
            return

        # count true positive, false positive, and false negative
        tp, fp, fn = 0, 0, 0
        for nfaces_gt, nfaces_pr in zip(self.all_nfaces_gt, self.all_nfaces_pr):
            if nfaces_gt == nfaces_pr:
                tp += nfaces_gt
            elif nfaces_gt > nfaces_pr:
                tp += nfaces_pr
                fn += nfaces_gt - nfaces_pr
            else:
                tp += nfaces_gt
                fp += nfaces_pr - nfaces_gt
                        
        # Precision, recall, and F1 score for all the faces
        precision = tp / (tp + fp) if tp + fp > 0 else 1.0
        recall = tp / (tp + fn) if tp + fn > 0 else 1.0
        f1_score = 2 * precision * recall / (precision + recall)
        
        # false positives and false negatives per image
        nimg = len(self.all_nfaces_gt)
        fp_per_img = fp / nimg
        fn_per_img = fn / nimg
        
        # print the results
        print(f"precision: {precision:.3f}")
        print(f"recall: {recall:.3f}")
        print(f"f1_score: {f1_score:.3f}")
        print(f"False positives (wrong detections) per image: {fp_per_img:.1f}")
        print(f"False negatives (miss detections) per image: {fn_per_img:.1f}")


def main():
    """
    Main function to execute the model inference and metric collection.
    """
    # parse arguments
    args = parser.parse_args()
    
    # load model settings
    model_settings = ModelSettings()

    # setup net and model
    net = setup_RetinaFace_model_inference(model_settings)    

    # get the image paths    
    img_paths = get_file_paths(args.path)
    if len(img_paths) == 0:
        return

    # metric collector
    metric_collector = MetricCollector()

    # loop over images
    for img_path in img_paths:        
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)        
        if img_raw is None:
            print(f"[Warning] unable to read {img_path} (not an image file?)")
            continue
        
        # run model inference and get bbox and landmarks
        dets = run_RetinaFace_inference(net, model_settings, img_raw)
        
        # collect prediction result
        nfaces_gt = int(basename(img_path).split("_")[0])  # num of faces (ground truth)
        nfaces_pr = len(dets)  # num of faces (prediction)
        metric_collector.collect(nfaces_gt, nfaces_pr)

        # print prediction result
        print(f"{basename(img_path)}\t{nfaces_pr}")
        
    # print metrics
    if args.metric:
        metric_collector.calculate_and_print()
        

if __name__ == "__main__":
    main()
