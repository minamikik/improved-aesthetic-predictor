import os
import requests
import clip
import torch
import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image

class AestheticPredictor():
    def __init__(self, model_path = "sac+logos+ava1-l14-linearMSE.pth"):

        self.model_path = model_path
        self.model_path = os.path.abspath(os.path.join(self.model_path))
        if not os.path.exists(self.model_path):
            self.download_model()

        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load(self.model_path)   # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)
        self.model.to("cuda")
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model2, self.preprocess = clip.load("ViT-L/14", device=self.device)  #RN50x64

    def download_model(self):
        remote = "https://github.com/minamikik/improved-aesthetic-predictor/raw/main/models/"
        url = remote + self.model_name
        response = requests.get(url)
        if response.status_code == 200:
            with open(self.model_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception("Error downloading the model")
        return

    def predict(self, pil_image: Image):
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model2.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy() )
        prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        return prediction.item()

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

