from PIL import Image
import numpy as np 

from robustness.datasets import ImageNet 
from robustness.model_utils import make_and_restore_model
import torch
import matplotlib.pyplot as plt

ds = ImageNet('/tmp')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path='/home/siddhant/Downloads/imagenet_l2_3_0.pt')
model.eval()

img = np.asarray(Image.open('/home/siddhant/CMU/robustness_applications/sample_inputs/img_bear.jpg').resize((224, 224)))
img = img/254. 
img = np.transpose(img, (2, 0, 1))

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

img_var = torch.tensor(img, dtype=torch.float)[None, :]
img = img_var.clone().detach().cpu().numpy()
img = img[0]

img = img.transpose((1, 2, 0))
img *= 255
img[img < 0] = 0
img = np.uint8(img)
plt.imshow(img)
plt.show()

ATTACK_EPS = 40
ATTACK_STEPSIZE = 1
ATTACK_STEPS = 200
NUM_WORKERS = 8
BATCH_SIZE = 10

def generation_loss(mod, inp, targ):
    op = mod(inp)
    loss = torch.nn.CrossEntropyLoss(reduction='none')(op, targ)
    return loss, None

kwargs = {
    'custom_loss' : generation_loss,
    'constraint':'2',
    'eps': ATTACK_EPS,
    'step_size': ATTACK_STEPSIZE,
    'iterations': ATTACK_STEPS,
    'targeted': True,
    'do_tqdm': True
}
target = torch.tensor([386], dtype=torch.long).cuda()

_, im_adv = model(img_var, target, make_adv=True, **kwargs)

img = im_adv.clone().detach().cpu().numpy()
# print(img.shape)
img = img[0]

img = img.transpose((1, 2, 0))
img *= 255
img[img < 0] = 0
img = np.uint8(img)
plt.imshow(img)
plt.show()