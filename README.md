# Project Name: Interpretation Adversarial Attack on Explainable AI using Electronic Health Records (EHR)

## 🚀 Project Overview
Welcome to the Interpretation Adversarial Attack project! This project is all about crafting a unique kind of attack known as the "Interpretation Adversarial Attack" on health data, specifically Electronic Health Records (EHRs). Our goal is to perturb the explanations provided by the RETAIN model to influence its predictions for a particular class within EHR input data. This project represents a pioneering effort in implementing such an attack in the medical domain. The RETAIN model, an attention-based model with two levels of LSTMs, is used for this purpose and has been customized to work seamlessly with EHR data.

## ⚙️ Prerequisites
Before diving into this project, make sure you have the following software and libraries installed:

- Python (>= 3.8)
- PyTorch (>= 1.10.0)
- NumPy (>= 1.21.0)
- Matplotlib (>= 3.4.3)
- scikit-learn (>= 0.24.2)


## 🏃‍♀️ How to Run

### Usage
To get started, you can modify the settings in the `settings.py` file to fine-tune the project parameters.

### Train the Model
Train the RETAIN model and test it with the following commands:

```bash
python train_RETAIN.py --data_path 'path_to_the_EHR_data_file' --save 'path to save the model checkpoints'
python test_RETAIN.py --save 'path to save the model checkpoints'
For training, you have the flexibility to adjust parameters such as learning rate (--lr), number of epochs (--epochs), and other settings found in settings.py.
```

### Run the Attack
To execute the adversarial attack, use the following command:

```bash
python run_attack.py --attacktype 'type of the attack based on the loss function. It can be "1": original attack, "2": KL divergence attack, 3: confident attack'
```

## 📁 File Structure
utils.py: Contains utility functions for data processing and handling.
settings.py: Configuration settings for the project.
train_RETAIN.py: Script to train the RETAIN model.
test_RETAIN.py: Script to test the RETAIN model.
retain.py: Implementation of the RETAIN model.
run_attack.py: Executes the adversarial attack by calling modules from interpretation_attack.py.
interpretation_attack.py: Houses our interpretation attack logic.
attack_core.py: Superclass for our attack.

## 👩‍💻 Author
This project is maintained by Fereshteh Razmi.

### ✉️ Contact
Feel free to reach out to me via email at: fe.razmi@gmail.com.

If you need the third part of the README or any further modifications, please let me know.





