from typing import Union
from pathlib import Path
import hashlib
from _hashlib import HASH as Hash
import random
import numpy as np
import torch
import os

import torch.nn.functional as F
from tqdm import tqdm
import mlflow

def train_model(student, train_dataloader, criterion, optimizer, epochs, device, teacher = None):
    student.train()
    student.to(device)
    student_name = "independent_student"
    if teacher:
        student_name = "student"
        teacher.eval()
        teacher.to(device)
        
    for epoch in range(epochs):
        running_loss = 0.0
        sampleNum = 0
        currentLoss = 0

        for inputs, labels in tqdm(train_dataloader, leave = True, position = 0):
        # for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = F.one_hot(labels, num_classes=1000).float()

            # Zero the gradients 
            optimizer.zero_grad()

            if teacher:
                teacher_predictions = teacher(inputs)
            student_predictions = student(inputs)

            if teacher:
                loss = criterion(student_predictions, labels, teacher_predictions, 0.5, 0.5)
            else:
                loss = criterion(independent_student_predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            currentLoss += loss.item()

            # if sampleNum % 100 == 0 and sampleNum > 0: # write this to a file somewhere else so tqdm doesnt mess up
            #     print("loss: ", str(currentLoss/100))
            #     print("total loss: ", str(running_loss/sampleNum))
            #     currentLoss = 0
            sampleNum += 1
            # print(str(running_loss) + " ", end = '')
            # break

        average_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss:.4f}')

        # save training loss in mlflow
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_metric(student_name + "_training_loss", average_loss)

    with mlflow.start_run(run_id=run_id) as run:
        if teacher:
            mlflow.pytorch.log_model(
                pytorch_model=teacher.to("cpu"),
                artifact_path="teacher",)

        mlflow.pytorch.log_model(
            pytorch_model=student.to("cpu"),
            artifact_path=student_name)

def train_models(components, teacher, train_dataloader, epochs, device, run_id):
    def handleModel(mod):
        mod.train()
        mod.to(device)
    def handleTeacher(teach):
        teacher.eval()
        teacher.to(device)
    
    def mlThing(name, inputs, loss_cntr):
        components[name]["opt"].zero_grad()
        predictions = components[name]["model"](inputs)
        if "ind" in name:
            loss = components[name]["criterion"](predictions, labels)
        else:
            teacher_predictions = teacher(inputs)
            loss = components[name]["criterion"](predictions, labels, teacher_predictions, 0.5, 0.5)
        loss.backward()
        # if loss_cntr == 0:
        #     components[name]["previous_loss"] = loss.item()
        # elif loss_cntr == 1:
        #     components[name]["previous_loss"] = (components[name]["previous_loss"] - loss.item)
        # else:
        #     components[name]["previous_loss"] = (components[name]["previous_loss"] - loss.item)
        components[name]["previous_loss"] = loss.item()
        components[name]["opt"].step()
        components[name]["running_loss"] += loss.item()
        return loss

    epoch_print = 4000

    for name in components:
        handleModel(components[name]["model"])
    handleTeacher(teacher)
        
    for epoch in range(epochs):
        "Reset loss"
        for student in components:
            components[student]["running_loss"] = 0
        loss_cntr = 0
        indprevious_loss = 0.0
        stdprevious_loss = 0.0
        
        "Begin training"
        for inputs, labels in tqdm(train_dataloader, leave = True, position = 0):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = F.one_hot(labels, num_classes=1000).float()

            for student in components:
                loss = mlThing(student, inputs, loss_cntr)
                
                if loss_cntr > 0 and loss_cntr % epoch_print == 0:
                    with open("test.txt", "a") as myfile:
                        # myfile.write("appended text")
                        previous_loss = components[student]['previous_loss']
                        running_loss = components[student]['running_loss']
                        myfile.write("Current loss: " + str(loss.item()))
                        myfile.write("\n")
                        myfile.write("Average loss over last something iterations: " + str(running_loss/epoch_print))
                        # myfile.write("\n")
                        # myfile.write("Avg delta loss per batch: " + str(previous_loss - loss.item()))
                        myfile.write("\n")
                        myfile.write("Average Loss delta: " + str(running_loss/epoch_print - previous_loss))
                        
                        myfile.write("\n")
                        myfile.write("\n")
                        components['student']['previous_loss'] = 0.0
            loss_cntr += 1
                

        """ End of the epoch """
        for student in components:
            avg_loss = components[student]["running_loss"] / len(train_dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], {student}: {avg_loss:.4f}')

            # save training loss in mlflow
            with mlflow.start_run(run_id=run_id) as run:
                mlflow.log_metric(student, avg_loss)


    with mlflow.start_run(run_id=run_id) as run:
        for student in components:
            mlflow.pytorch.log_model(
                pytorch_model=components[student]["model"].to("cpu"),
                artifact_path=student)

        mlflow.pytorch.log_model(
            pytorch_model=teacher.to("cpu"),
            artifact_path="teacher",)

                
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# to be called my md5_dir
def md5_update_from_dir(directory: Union[str, Path], hash: Hash) -> Hash:
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            hash = md5_update_from_file(path, hash)
        elif path.is_dir():
            hash = md5_update_from_dir(path, hash)
    return hash

# given a directory of data, return the hash
def md5_dir(directory: Union[str, Path]) -> str:
    return str(md5_update_from_dir(directory, hashlib.md5()).hexdigest())

def md5_update_from_file(filename: Union[str, Path], hash: Hash) -> Hash:
    assert Path(filename).is_file()
    with open(str(filename), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

# check if parameters are all the same
def check_two_models_have_same_weights(model1, model2):
    # Get state dictionaries of the models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Check if keys are the same
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    # Check if values are the same for each key
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


