# Traffic Sign Recognition with a CNN

A convolutional neural network that classifies photos of traffic signs into **43 categories**, trained on the German Traffic Sign Recognition Benchmark (GTSRB).

**Result: 99.1–99.7% test accuracy**, with a model kept small enough to stay fast at inference.

## The model

The architecture is inspired by VGG: stacked 3×3 convolutions, with the number of filters doubling as the network gets deeper.

```
Rescale (0–1)
Conv 32 → BatchNorm → Conv 32 → BatchNorm → MaxPool → Dropout
Conv 64 → BatchNorm → Conv 64 → BatchNorm → MaxPool → Dropout
Conv 128 → BatchNorm → MaxPool → Dropout
Flatten → Dense 256 → Dropout → Dense 43 (softmax)
```

## What I learned from experimenting

I tried a lot of variations before settling on this one. Some takeaways:

- **Batch normalization made the biggest difference.** On the final model it moved accuracy from 99.0% to 99.7% and cut loss from 0.036 to 0.010. On simpler models the gap was even larger.
- **Two 3×3 convolutions before each pooling layer** beat single, larger kernels. They give a 5×5 receptive field with fewer parameters and an extra non-linearity.
- **More layers didn't help.** Bigger models scored higher on training data but worse on validation, so they were overfitting.
- **Bicubic resizing** of the input images gave slightly more consistent accuracy than other interpolation methods.
- **Things that didn't help:** data augmentation, L1/L2 regularization, learning-rate schedules, custom initializers, and gradient clipping. Dropout alone was enough to prevent overfitting.

The full experiment log is in [`NOTES.md`](NOTES.md).

## Run it

```bash
pip install -r requirements.txt
python traffic.py gtsrb-small            # train and evaluate
python traffic.py gtsrb-small model.h5   # also save the trained model
```

The repo includes a small sample of the dataset in `gtsrb-small/`. For the full results, download the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html) and pass its folder instead. The data needs one subfolder per class, named `0` to `42`.

## Built with

Python · TensorFlow / Keras · OpenCV · scikit-learn
