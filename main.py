import random
from engine import Value
from nn import MLP

data = [
    ([2.0, 3.0], 1.0),
    ([3.0, -1.0], -1.0),
    ([1.0, 1.0], 1.0),
    ([2.0, -2.0], -1.0)
]


model = MLP(2, [4, 1])


epochs = 100 
learning_rate = 0.01

for k in range(epochs):
    
    total_loss = Value(0)
    for x, y in data:
        x = [Value(xi) for xi in x]  
        y_pred = model(x)  
        loss = (y_pred - Value(y)) ** 2  
        total_loss += loss
    
    model.zero_grad()  
    total_loss.backward()  
    
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    

    print(k, total_loss.data)
