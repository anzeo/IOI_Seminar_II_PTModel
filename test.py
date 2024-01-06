import json
import numpy as np
import torch
from util import pre_process_landmark


model = torch.load("out_models/model.pt")
with torch.no_grad():
    model.eval()
    json_data = json.load(open("data/test_data_tmp/data.json"))
    X_dataset = np.asarray(json_data["data"])

    X_dataset = np.array([pre_process_landmark(x) for x in X_dataset])

    tensor_x = torch.Tensor(X_dataset)

    prediction = model(tensor_x)
    for i in prediction:
        print(np.rint(i))
