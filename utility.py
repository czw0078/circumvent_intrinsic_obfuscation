import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self): # TODO change the name from conv1 to cnn1
        super(LeNet, self).__init__() # can be just super() in python 3
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,  # Number of channels in the input image      
                out_channels=16, # Number of channels produced by the convolution           
                kernel_size=5,              
                stride=1,                   
                padding=2, # half of the kernel size 5 except middle 1       
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),       
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    # do not return x for visualization

def network():
    net = torch.load('./data/LeNet.pt')
    net.eval()
    return net

def samples(x=None,adversarial=True):
    lbl100=torch.load('./data/samples/lbl100.pt')
    if x != None:
        return [(x, lbl100)] 
    elif not adversarial:
        img100=torch.load('./data/samples/img100.pt')
        return [(img100, lbl100)]
    else:
        adv100=torch.load('./data/samples/adv100.pt')
        return [(adv100, lbl100)]

def identity(x, edge):
    return x

def evaluate(net, loader, defense=identity, edge=0.3, device=torch.device("cpu")):
    with torch.no_grad(): 
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            test_output = net(defense(images, edge)) # output are in loader batch size
            # max work on second dimention of batch 
            # variable to tensorsqueeze for python 2
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # print(pred_y) print(labels)
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / float(total)
        print('Check accuracy of the model:', accuracy)

def trimmer(x, edge=0.3): # 0 to 0.5
    """
    >>> import torch
    >>> a = torch.Tensor([[0.7810, 0.2048, 0.2540],[0.4569, 0.3009, 0.1701]])
    >>> trimmer(a)
    tensor([[1.0000, 0.0000, 0.0000],
            [0.4569, 0.3009, 0.0000]])
    """

    low = edge
    high = 1 - edge
    y = torch.clone(x)

    select_high = y >= high
    select_low = y <= low

    y[select_high] = 1
    y[select_low] = 0

    return y

if __name__=="__main__":
    import doctest
    doctest.testmod()
