#  Deep Learning Assignment – CNN from Scratch

This project implements a **custom CNN model** trained on the iNaturalist dataset for a 10-class image classification task. It includes training, hyperparameter tuning using Weights & Biases (W&B), and model evaluation. **No transfer learning was used**—the model is built and trained from scratch.

---

##  Project Structure

### 1.  Output Size Computation
- A helper function was created to **compute the output size** after each convolution and pooling operation.
- Ensures smooth dimension flow from convolutional layers to fully connected layers.

### 2.  Custom CNN Model
- `ConvNetModel` class:
  - **5 Convolutional Blocks**: Each follows the structure Conv → Activation → MaxPool
  - Supports **custom filter sizes, kernel sizes, activation functions**, and pooling strategy
  - Optionally includes **Batch Normalization** and **Dropout**
  - Ends with **a fully connected layer** and **output layer** with 10 output neurons (for 10 classes)

---

##  Data Handling

### `data_load()`
- Loads data and splits it into training and validation sets
- Parameters:
  - `batch_size`
  - `val_split`
  - `data_augmentation` (True/False)

### `load_train_data()`
- Simplified wrapper to load training data with or without data augmentation

---

##  Training and Sweeps

### `model_train_val()`
- Trains the CNN model using:
  - Loss: CrossEntropyLoss
  - Optimizer: Adam
  - Default Learning Rate: 0.001

### `sweep()`
- Uses **W&B Bayesian Optimization sweeps** to tune hyperparameters:
  - Kernel sizes across 5 layers
  - Dropout values (0.2, 0.3)
  - Activation functions (`relu`, `mish`, `silu`, `gelu`)
  - Dense layer sizes (128, 256)
  - Batch normalization (True/False)
  - Filter organizations (various combinations)
  - Data augmentation (True/False)
  - Epochs (5, 10, 15)
- Efficient way to explore multiple model variants and configurations

---

##  Final Evaluation

### `test_evaluation_model()`
- Loads best weights from training
- Evaluates on test set
- Computes overall and class-wise accuracy and loss

### `plot_predictions()`
- Visualizes predictions
- Shows actual vs predicted labels for sample images

---

##  Additional Features

- **Batch Normalization** to stabilize and speed up training
- **Dropout** to prevent overfitting
- **W&B integration** for tracking metrics, visualizing loss/accuracy curves, and managing sweep results

---

## 📈 Results & Observations

- Training from scratch yielded strong performance
- Sweeps helped identify best combination of dropout, filter arrangement, kernel size, and activations
- **Data augmentation** positively impacted validation accuracy
- Best models generalized well to test set

---

##  Dependencies

- Python 3.7+
- torch, torchvision
- wandb
- matplotlib, numpy

---

## ▶ Run

```python
# To train
main()

# To evaluate test set with best model
test_evaluation_model(model_path='best_model.pth')

# To plot predictions
plot_predictions(model, test_loader)
```

---

