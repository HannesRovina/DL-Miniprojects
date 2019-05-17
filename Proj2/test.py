from torch import empty
from torch import set_grad_enabled
import src.modules as module
from src.optimizer import SGD, NesterovSGD
from src.functional import generate_disc_set, train

# Disable autograd
set_grad_enabled(False)

# Generating the dataset
N = 1000
test_input, test_target = generate_disc_set(N, batch_size=50)
train_input, train_target = generate_disc_set(N, batch_size=50)

# Building the model
# 3 hidden layers, 25 hidden nodes each

model = module.Sequential(('fc1',module.Linear(2,25)),('relu1',module.ReLU()),
                          ('fc2',module.Linear(25,25)),('relu2',module.ReLU()),
                          ('fc3',module.Linear(25,25)),('relu3',module.ReLU()),
                          ('fc4',module.Linear(25,2)),('tanh1',module.Tanh()))
criterion = module.MSELoss()

# Training 30 epochs, lr=0.005
train_epochs = 30
lr = 0.005
optim = SGD(model.parameters(), lr=lr)
print("Training with lr = {0:.02e} over {1} epochs".format(lr, train_epochs))

final_train, final_test = train(train_input, train_target, test_input, test_target, 
                                model, optim, criterion, epochs=train_epochs) 
print("================================================")
print("Final performance after {0} epochs".format(train_epochs))
print("------------------------------------------------")
print("Train loss {0:.02f} ++++ Test loss {1:.02f}".format(final_train[0], final_test[0]))    
print("Train accuracy {0:.02f}% ++++ Test accuracy {1:.02f}%".format(final_train[1], final_test[1]))
  