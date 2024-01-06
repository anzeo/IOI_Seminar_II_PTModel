import numpy as np
import torch

from model.model import HandLandmarkModel


def pre_process_landmark(landmark):
    tmp_landmark = np.array(landmark)

    base_x, base_y, base_z = tmp_landmark[0, 0], tmp_landmark[0, 1], tmp_landmark[0, 2]

    tmp_landmark[:, 0] -= base_x
    tmp_landmark[:, 1] -= base_y
    tmp_landmark[:, 2] -= base_z

    return tmp_landmark.flatten()


def convert_pt_model_to_onnx(model_name):
    pytorch_model = HandLandmarkModel(63, 2)
    pytorch_model.load_state_dict(torch.load("out_models/" + model_name + ".pt"))
    pytorch_model.eval()
    dummy_input = torch.zeros(1, 63)
    torch.onnx.export(pytorch_model, dummy_input, "out_models/" + model_name + ".onnx", verbose=True)
