from torch.utils.data import Dataset, DataLoader, random_split
from torchaudio import load_wav
from torchaudio.transforms import MFCC
import os
import re
import torch


class FSDD(Dataset):
    def __init__(self, root: str, training: bool=True, max_length: int=2500):
        self.data = []
        self.labels = []
        self.transform = MFCC(sample_rate=8000)

        for filename in os.listdir(root):
            info = re.split(r'[_.]', filename)
            if (training and int(info[2]) > 4) or (not training and int(info[2]) < 5):
                filepath = root + filename
                input_audio = self.transform(load_wav(filepath)[0])[0, :, :max_length]
                if input_audio.shape[1] < max_length:
                    input_audio = torch.cat([input_audio, torch.zeros((40, max_length - input_audio.shape[1]))], dim=1)
                self.data.append(input_audio)
                self.labels.append(int(info[0]))

    def __getitem__(self, i: int):
        return self.data[i], self.labels[i]

    def __len__(self):
        return len(self.data)


train_val_dataset = FSDD("free-spoken-digit-dataset-master/recordings/", True)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_data, val_data = random_split(train_val_dataset, lengths=[train_size, val_size])

test_data = FSDD("free-spoken-digit-dataset-master/recordings/", False)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

from torch import nn
from nntoolbox.components import AveragePool
from torch.optim import Adam

model = nn.Sequential(
    nn.Conv1d(40, 64, 3),
    nn.ReLU(),
    nn.Conv1d(64, 128, 3),
    nn.ReLU(),
    nn.Conv1d(128, 10, 3),
    AveragePool(dim=2)
)

from nntoolbox.learner import SupervisedLearner
from nntoolbox.callbacks import *
from nntoolbox.metrics import *
learner = SupervisedLearner(train_loader, val_loader, model=model, criterion=nn.CrossEntropyLoss(), optimizer=Adam(model.parameters()))
callbacks = [
    ToDeviceCallback(),
    LossLogger(),
    ModelCheckpoint(learner=learner, filepath="weights/model.pt", monitor='accuracy', mode='max'),
]

metrics = {
    "accuracy": Accuracy(),
    "loss": Loss()
}

final = learner.learn(
    n_epoch=500,
    callbacks=callbacks,
    metrics=metrics,
    final_metric='accuracy'
)


print(final)
