# Adversarial-Attacks-and-Defenses-in-Deep-Learning
This project explores adversarial attacks and defenses in the context of deep learning models, particularly Convolutional Neural Networks (CNNs). Adversarial attacks attempt to mislead machine learning models by introducing small perturbations to the input data. The project implements several types of attacks and defenses:
Noise Attack: Adds random noise to the input to observe its effect on the model.
Carlini-Wagner (CW) Attack: An adversarial attack method that seeks to minimize the perturbation while still misleading the classifier.
Fast Gradient Sign Method (FGSM) Attack: A fast, one-step attack that adds perturbations in the direction of the gradient of the loss with respect to the input.
Projected Gradient Descent (PGD) Attack: A multi-step variant of FGSM that iteratively adjusts the perturbations within a defined range.
Defensive mechanisms, such as adversarial training, are implemented to make models more robust against these adversarial attacks.

Structure
1. Convolutional Neural Network (CNN) Definition
The project uses a simple CNN architecture for image classification tasks:

Two convolutional layers with ReLU activation and max pooling.
Three fully connected layers to perform the final classification.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
2. Adversarial Attacks
A. Noise Attack
Adds random noise to the input image and evaluates the model's robustness.
Simple yet effective for understanding the model's sensitivity to small changes in the input.
B. Carlini-Wagner (CW) Attack
A more advanced adversarial attack aimed at finding the smallest possible perturbation to fool the model.
Uses L2 loss combined with the classification error to minimize the perturbation size.
Implemented using gradient-based optimization to iteratively update the perturbation.

def cw_attack(model, image, label, device, kappa=0, c=0.1, lr=0.01, max_iter=100):
    # Initialize the perturbation and perform attack

C. Fast Gradient Sign Method (FGSM) Attack
A one-step attack where perturbations are added in the direction of the gradient of the loss with respect to the input.
Efficient and fast but may not always generate the smallest perturbation.


def fgsm_attack(model, images, labels, epsilon):
    # Add perturbation to images based on gradients
D. Projected Gradient Descent (PGD) Attack
Iterative version of FGSM, which applies small perturbations in multiple steps while ensuring the perturbation stays within a specified bound.
PGD is considered one of the strongest first-order adversarial attacks.


def pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    # Iteratively apply small perturbations
    
3. Adversarial Defenses
Adversarial Training
The model is trained using both clean and adversarial examples to improve its robustness.
Helps the model generalize better and resist adversarial perturbations.
4. Performance Evaluation
The training and evaluation loops measure the performance of the CNN on both clean and adversarially perturbed inputs.
Metrics such as accuracy and loss are logged for analysis.


def measure_performance(algorithm, *args, **kwargs):
    # Measure the execution time and performance of each attack
Requirements
Python 3.x
PyTorch
torchvision for dataset handling
numpy and matplotlib for data manipulation and visualization
How to Run
Clone the repository:


git clone https://github.com/yourusername/adversarial-attacks-defense.git
Install the required dependencies:


pip install -r requirements.txt
Run the notebook or script to execute the adversarial attacks:



python run_attacks.py
To visualize the results: Open the Jupyter notebooks to explore the effect of the adversarial attacks on the trained models.

Example Usage
Noise Attack Example:

# Apply noise to a test image and evaluate the model's response
noise_image = add_noise(image)
output = model(noise_image)
CW Attack Example:

# Perform CW attack
perturbed_image = cw_attack(model, image, label, device)

Results
Detailed results on how different adversarial attacks affect model performance.
Comparisons between different attack methods and the robustness provided by adversarial training.
License
This project is licensed under the MIT License.
