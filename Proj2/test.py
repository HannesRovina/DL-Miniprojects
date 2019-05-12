from torch import empty
from torch import set_grad_enabled
import src.modules as module
from src.optimizer import SGD
from src.functional import generate_disc_set

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
                          ('fc3',module.Linear(25,2)))

optim = SGD(model.parameters(), lr=0.01)
criterion = module.LossMSE()

def train(train_input, train_target, test_input, test_target, model, optim, criterion, epochs=20):
    for i in range(epochs):
        print("Epoch {0}".format(i))
        loss_train = 0
        acc_train = 0
        loss_test = 0
        acc_test = 0
        
        for x, y in zip(train_input, train_target):
            model.train()
            pred = model(x)
            loss_train += criterion(pred, y) 
            
            # Set gradient to zero
            optim.zero_grad()
            grad_loss = criterion.backward(pred, y)
            model.backward(grad_loss)
            optim.step()
            
        
        for test_x, test_y in zip(test_input, test_target):
            model.test()
            loss_test += criterion(model(test_x), test_y)

        print("Train loss {0:.02f} ++++ Test loss {1:.02f}".format(loss_train/len(train_input), loss_test/len(test_input)))
        
train(train_input, train_target, test_input, test_target, model, optim, criterion, epochs=15)       