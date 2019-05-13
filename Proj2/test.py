from torch import empty
from torch import set_grad_enabled
import src.modules as module
from src.optimizer import SGD
from src.functional import generate_disc_set, train, tune_lr

# Disable autograd
set_grad_enabled(False)

# Generating the dataset
N = 1000
train_input, train_target = generate_disc_set(N, batch_size=50)
test_input, test_target = generate_disc_set(N, batch_size=50)

# Building the model
# 3 hidden layers, 25 hidden nodes each

model = module.Sequential(('fc1',module.Linear(2,25)),('relu1',module.ReLU()),
                          ('fc2',module.Linear(25,25)),('relu2',module.ReLU()),
                          ('fc3',module.Linear(25,2)),('tanh1',module.Tanh()))


criterion = module.MSELoss()

lambdas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
best_lr = tune_lr(lambdas, train_input, train_target, test_input, test_target, model, SGD, criterion, epochs=25)

optim = SGD(model.parameters(), lr=best_lr)
train_epochs = 50
print("Training with lr = {0:.02e} over {1} epochs".format(best_lr, train_epochs))
train(train_input, train_target, test_input, test_target, model, optim, criterion, epochs=50)       