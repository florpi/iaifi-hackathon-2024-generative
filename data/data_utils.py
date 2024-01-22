
import numpy as np
import torch 
import torch.nn.functional as F

class ToRGB:
    '''takes in batched image of size (batch_size, npix, npix, nchannel), 
    and converts from native telescope image scaling to rgb'''

    def __init__(self, scales=None, m=0.03, Q=20., bands=['g', 'r', 'z']):
        rgb_scales = {'u': (2,1.5),
                      'g': (2,6.0),
                      'r': (1,3.4),
                      'i': (0,1.0),
                      'z': (0,2.2)}
        if scales is not None:
            rgb_scales.update(scales)
            
        self.rgb_scales = rgb_scales
        self.m = m
        self.Q = Q
        self.bands = bands
        self.axes, self.scales = zip(*[rgb_scales[bands[i]] for i in range(len(bands))])
        
        # rearrange scales to correspond to image channels after swapping
        self.scales = [self.scales[i] for i in self.axes]

    def __call__(self, image):
        # Assuming image is of shape [batch_size, npix, npix, nchannel]
        image = image.permute(0, 3, 1, 2)  # Change to [batch_size, nchannel, npix, npix]
        image = image[:, self.axes]
        image = image.permute(0, 2, 3, 1)  # Change back to [batch_size, npix, npix, nchannel]
        scales = torch.tensor(self.scales, dtype=torch.float32).to(image.device)

        I = torch.sum(torch.clamp(image * scales + self.m, min=0), dim=-1) / len(self.bands)
        
        fI = torch.arcsinh(self.Q * I) / np.sqrt(self.Q)
        I += (I == 0.) * 1e-6
        
        image = (image * scales + self.m) * (fI / I).unsqueeze(-1)
        image = torch.clamp(image, 0, 1)

        return 2. * image.permute(0, 3, 1, 2) - 1.  # Change to [batch_size, nchanne1, ntx, npix]