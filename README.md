# Accentuate class labels from a smoothed classifier

## Smoothed classifiers

For a given base network $f$, a smoothed classifier $f_{\sigma}$  defined as the classifier when queried with an input $x$, returns the expected scores of the base classifier $f$ when the input, $x$, is perturbed with an isotropic Gaussian noise, 

$$f_{\sigma} = \mathop{\mathbb{E}}_{\mathcal{E} \sim \mathcal{N}(0, \sigma^{2}\mathcal{I})} [ f(x + \mathcal{E}) ]$$.

Hence, the predicted class of a smoothed classifier is the expected class for the input in its $l2$ neighbourhood.

## Image generation training objective for a smoothed classifier
Let $\mathcal{L}$ be the loss function (Cross Entropy or Target class Maximum) for maximising the expected score of a class. We minimize this loss using Projected Gradient Descent to update the input features(and keeping the model parameters fixed). 

$$\mathcal{L}_{CE}(f_{\sigma}, x, c) = log(f_{\sigma}(x)_{c})$$

$$\implies \mathcal{L}_{CE}(f_{\sigma}, x, c) = log \big( \mathbb{E}_{\mathcal{E} \sim \mathcal{N}(0, \sigma^{2}\mathcal{I})} [ f(x + \mathcal{E})_{c} ]\big)$$

$$\implies \mathcal{L}_{CE}(f_{\sigma}, x, c) = \mathbb{E}_{\mathcal{E} \sim \mathcal{N}(0, \sigma^{2}\mathcal{I})} [log ( f(x + \mathcal{E})_{c} )]$$\\

Similarly, $L_{TCM}$ can be defined as, 

$$\mathcal{L}_{TCM}(f_{\sigma}, x, c) = \mathbb{E}_{\mathcal{E} \sim \mathcal{N}(0, \sigma^{2}\mathcal{I})} [logits_{f}(x + \mathcal{E})_{c} ]$$.

The expectation can be evaluated directly, hence a Monte Carlo estimate needs to be taken.

## Implementation steps

The base classifier $f$ needs to be trained to recognise the Gaussian noise that is added to the image. This is done by augmenting the dataset with Gaussian noise. 

### Training steps

To train such a classifier, run the following, 

```
python code/train_classifier.py --dataset imagenet --arch resnet50 --out_dir $OUT_DIR --noise_sd 0.5
```

The other hyperparameters for training can also be set up through command line arguments.

### Certify

To certify the robustness (prediction and radius) of the smoothed classifier, 


### Image generation


## Adversarial training

## Image generation through adversarial trained classifiers