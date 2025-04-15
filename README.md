# CNN
CNN based image classifiers using a subset of the iNaturalist dataset.

# Build a CNN with:
- 5 Conv → Activation → MaxPool blocks
- Customizable dense & output layers (10 classes)
- Flexible filters, kernel sizes, activations, and neurons
- Compute total parameters & operations (based on m, k×k, n)
# Train on iNaturalist:
- Use 80/20 train-validation split (balanced by class)
# Apply WandB sweeps for hyperparameter tuning:
- Filters, activations, dropout, batch norm, etc.
# Include:
- Accuracy vs experiments plot
- Parallel coordinates & correlation summary
- Report test accuracy
- Show results in a creative 10×3 prediction grid.
# Part B : Fine-tuning a pre-trained model
- Loads a pre-trained model (e.g., ResNet50, VGG, EfficientNetV2, ViT) from torchvision, trained on ImageNet.
- Fine-tunes it on iNaturalist dataset instead of training from scratch.
