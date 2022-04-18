import glob
import pathlib
import torchvision.transforms as trans
from django.shortcuts import render,HttpResponse
import pickle
import torch
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn



class ConvNet (nn.Module): 
    def __init__ (self,num_classes=2):
        super (ConvNet, self).__init__()

        #((w-f+2P)/s) +1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features =12)
        self.relu1 = nn.ReLU()
       
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20, kernel_size=3, stride=1,padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32, kernel_size=3, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(num_features =32)
        self.relu3 = nn.ReLU()
        
        
        self.fc = nn.Linear(in_features = 32*75*75,out_features = num_classes)
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
            
        output = self.pool(output)
            
        output = self.conv2(output)
        output = self.relu2(output)
            
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
            
        output = output.view(-1,32*75*75)
            
        output = self.fc(output)
        return output
            

     

def prediction(img_path,transformer,classes):
    device = torch.device("cpu")
    checkpoint = torch.load('C:/Users/Ram/Desktop/Django/finalproject/myapp/model/best_checkint.model',map_location='cpu')
    
    model = ConvNet(num_classes = 2)
    model.load_state_dict(checkpoint)
    model.eval()
    image = Image.open(img_path)
    image_tensor = transformer(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    
        
    input = Variable(image_tensor)
    
    output = model(input)
    
    index = output.data.numpy().argmax()
    
    pred = classes[index]
    return pred




def pred(request):
    pred_path = 'c:/Users/Ram/Desktop/Dataset_Test/*'
    train_path = 'c:/Users/Ram/Desktop/Datasets'



    root = pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

    transformer = trans.Compose([
    trans.Resize((150,150)),
    trans.RandomHorizontalFlip(),
    trans.ToTensor(),
    trans.Normalize([0.5,0.5,0.5],
                        [0.5,0.5,0.5])
])  
    pred_dict = {}

    for i in glob.glob(pred_path):
        pred_dict[i[i.rfind('/*')+1:]]=prediction(i,transformer,classes)
    
    pred_output = pred_dict.get('c:/Users/Ram/Desktop/Dataset_Test\\1194960-pothole.jpg')
    a = {"pred_output":pred_output}
    
    if pred_output == 'pothole':
        sql = "insert into pothole "

    return render(request,'pred.html',a)



def index(request):
    return render(request, 'index.html')
