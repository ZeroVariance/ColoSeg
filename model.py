import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_mean = [0.76458206, 0.60492328, 0.7896577]  
dataset_std = [0.18823201, 0.24406966, 0.14710178]

class CRF(nn.Module):
    def __init__(self, num_iterations=5):
        super(CRF, self).__init__()
        self.num_iterations = num_iterations

    def forward(self, image, mask):
        # Convert image and mask to numpy arrays
        image_np = image.detach().cpu().numpy().squeeze()
        mask_np = mask.detach().cpu().numpy().squeeze()

        # Convert mask to one-hot encoding
        mask_one_hot = np.zeros((2, mask_np.shape[0], mask_np.shape[1]))
        mask_one_hot[0, :, :] = mask_np == 0
        mask_one_hot[1, :, :] = mask_np == 1

        # Set up dense CRF parameters
        d = dcrf.DenseCRF2D(image_np.shape[1], image_np.shape[0], 2)
        unary = np.array([1 - mask_one_hot[0], mask_one_hot[0]])
        unary = unary.reshape((2, -1))
        d.setUnaryEnergy(-np.log(unary))
        feats = np.concatenate([image_np[..., np.newaxis], image_np[..., np.newaxis]], axis=-1)
        feats = feats.reshape((-1, 2))
        d.addPairwiseGaussian(sxy=(3, 3), compat=3)
        d.addPairwiseBilateral(sxy=(50, 50), srgb=(20, 20, 20), rgbim=feats, compat=10)

        # Perform inference
        Q = d.inference(self.num_iterations)
        pred = np.argmax(np.array(Q), axis=0).reshape(mask_np.shape)
        return torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float().to(device)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_crf=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_crf = use_crf

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        if use_crf:
            self.crf = CRF(inference_type='softmax')
            
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        if self.use_crf:
            probs = F.softmax(logits, dim=1)
            images = x.detach().cpu().numpy().squeeze()
            probs = probs.detach().cpu().numpy().squeeze()
            unary = probs.transpose(1, 2, 0)
            unary = unary.reshape((-1, self.n_classes))
            unary = np.ascontiguousarray(unary)

            d = dcrf.DenseCRF(images.shape[1] * images.shape[0], self.n_classes)
            d.setUnaryEnergy(-np.log(unary))
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=images, compat=10)
            Q = d.inference(5)
            Q = np.array(Q).reshape((images.shape[0], images.shape[1], self.n_classes))
            logits = torch.tensor(Q.transpose(2, 0, 1)).unsqueeze(0).to(logits.device)
            
        return logits


def preprocess(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define the pre-processing pipeline
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # resize to 256x256 pixels
        transforms.ToTensor() # convert to tensor
    ])

    # Apply the pre-processing pipeline to the image
    image_tensor = transform(image)

    return image_tensor

model = UNet(n_channels=3, n_classes=7)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

image_path = 'example.jpg'

# predict_img(model, preprocess(image_path), device)
def predict(image_path, model):
    # Read the image
    image = cv2.imread(image_path)
    # Resize the image to the model's required input size
    image = cv2.resize(image, (256, 256))
    # Convert the image to the format expected by the model (PyTorch tensor)
    image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float().unsqueeze(0) / 255.0

    # Pass the image through the model to get the predicted segmentation
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).squeeze().cpu().numpy()

    # Convert the predicted segmentation to an RGB image for visualization
    preds_rgb = cv2.cvtColor((preds * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return preds_rgb




def plot_pred_images(input, output):
    fig = plt.figure(figsize=(15, 15))

    plt.subplot(1, 3, 1)
    plt.imshow(input.permute(1, 2, 0))
    plt.title('Image', color='white')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(output)
    plt.title('Segmentation', color='white')
    plt.axis('off')
    #plt.show()
    plt.savefig('static/images/output.png', bbox_inches='tight', transparent=True)

    return fig



# seg_mask = predict(image_path, model)
# plot_pred_images(preprocess(image_path), seg_mask)