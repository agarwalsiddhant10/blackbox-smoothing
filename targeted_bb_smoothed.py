from PIL import Image
import numpy as np 

from robustness.datasets import ImageNet 
from robustness.model_utils import make_and_restore_model
import torch
import matplotlib.pyplot as plt
from code.architectures import get_architecture
from robustness.attacker import AttackerModel
import torchvision

def random_perturb(x, eps):
    new_x = x + (torch.randn_like(x, device='cuda'))* 0.5
    return new_x

ds = ImageNet('/tmp')

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

classifier_path = '/home/siddhant/CMU/pretrained_models/imagenet_classifiers/resnet50/noise_0.50/checkpoint.pth.tar'
checkpoint = torch.load(classifier_path)
base_classifier = get_architecture('resnet50', 'imagenet')
base_classifier.load_state_dict(checkpoint['state_dict'])
print('loaded_basemodel')

model = AttackerModel(base_classifier, ds).cuda()

ATTACK_EPS = 100
ATTACK_STEPSIZE = 1.5
ATTACK_STEPS = 100
NUM_WORKERS = 8
BATCH_SIZE = 10

def generation_loss(mod, inp, targ):
    X = None
    tar = None
    for __ in range(0, 50):
        x = random_perturb(inp, 100)
        if X is None:
            X = x
            tar = targ
        else:
            X = torch.cat((X, x))
            tar = torch.cat((tar, targ))
    op = mod(X)
    print(torch.argmax(op, 1))
    loss = torch.nn.CrossEntropyLoss(reduction='none')(op, tar)
    loss = torch.mean(loss, 0, keepdim=True)
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
target = torch.tensor([1], dtype=torch.long).cuda()

_, im_adv = model(img_var, target, make_adv=True, should_normalize = False, **kwargs)

img1 = im_adv.clone().detach().cpu().numpy()
img1 = img1[0]

img1 = img1.transpose((1, 2, 0))
img1 *= 255
img1[img1 < 0] = 0
img1 = np.uint8(img1)
plt.imshow(img1)
plt.show()



