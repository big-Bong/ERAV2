# ERAV2
TSAI ERA V2 course

Each folder corresponds to a different session. This will be the root folder for the entire course. The latest session's assignment details will be on top. Please scroll down for older sessions.

## Session 6 (Advanced architectures)
### Part 1

> [Backpropagation Excel sheet](/Session6/VasudevChatterjee_ERAV2_Session6_Part1.xlsx?raw=true)

<br/>

#### **The Neural Network**
![Neural Network Architecture](/Session6/screenshots/ERAV2_Session6_NeuralNet.jpg?raw=true)

<br/>

For a detailed explanation of the above NN, and to look at the backpropagation calculations, please refer to the [excel sheet](/Session6/VasudevChatterjee_ERAV2_Session6_Part1.xlsx?raw=true), tagged here and above.

<br/>

#### **Learning rate vs Error graph**
<br/>

| LR      | Error Graph                              |
| ------- | -----------                              |
| `0.1`     | ![](/Session6/screenshots/lr_0.1.png) |
| `0.2`     | ![](/Session6/screenshots/lr_0.2.png) |
| `0.5`     | ![](/Session6/screenshots/lr_0.5.png) |
| `0.8`     | ![](/Session6/screenshots/lr_0.8.png) |
| `1`       | ![](/Session6/screenshots/lr_1.png)   |
| `2`       | ![](/Session6/screenshots/lr_2.png)   |

<br/>

### Part 2
> [Google Colab Notebook](/Session6/VasudevChatterjee_ERAV2_Session6_Part2_latestrun.ipynb)

The above notebook can be downloaded and then uploaded on google colab. The notebook is self-explanatory, and will run by executing each cell of the notebook at a time. The explanation for the code can also be found in the notebook.

The Session6 folder contains two ipynb notebooks. Both are the exact same code, just two different runs. Both have achieved 99.4% accuracy, however the "latestrun" notebook achieves that in the 4th last epoch. The other ipynb achieves that consistently in the last few epochs.

<br/>
<br/>

## Session 5 (PyTorch)

Session 5 contains the following files:
> 1. S5.ipynb: This is the main file from which the code will get executed. This is designed like a Jupyter/Google Colab notebook
> 2. models.py: This contains the actual neural network architecture
> 3. utils.py: Contains all utility functions related to training and testing the model. Getting the summary of the model, and transforming the data

#### Instructions for running code on Google Colab

> 1. Create your folder on gdrive and upload all 3 files in session 5 folder to your gdrive folder. All 3 files should be in the same directory.
> 2. Change path in cell number 4 and 5 to the path of your folder in gdrive
> 3. Run the S5 notebook one cell at a time