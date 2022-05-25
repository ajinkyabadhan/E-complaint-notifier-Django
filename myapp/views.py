import glob
import pathlib
import torchvision.transforms as trans
from django.shortcuts import render, HttpResponse, redirect
import pickle
import torch
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
# from .forms import imageUploadForm
from .forms import *
# from .models import imageUploadModel
from .models import *




class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()

        # ((w-f+2P)/s) +1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(in_features=32 * 75 * 75, out_features=num_classes)

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

        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)
        return output


def prediction(img_path, transformer, classes):
    device = torch.device("cpu")
    checkpoint = torch.load('c:/Users\Ram/Desktop/E-complaint-notifier-Django/myapp/model/classification2.model',
                            map_location='cpu')

    model = ConvNet(num_classes=3)
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


def imageUploadPage(request):
    context = {}
    if request.method == "POST":
        form = imageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            imageUploadPage.img1 = form.cleaned_data.get("Upload_Image")
            obj = imageUploadModel.objects.create(
                img=imageUploadPage.img1
            )
            obj.save()
            return redirect(pred)
    else:
        form = imageUploadForm()
    context['form'] = form
    return render(request, "upload.html", context)


pred_dict = {}
location_dict = {}
a = {}


def pred(request):
    pred_path = 'c:/Users/Ram/Desktop/E-complaint-notifier-Django/images/*'
    train_path = 'c:/Users/Ram/Desktop/E-complaint-notifier-Django/Datasets'

    root = pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

    transformer = trans.Compose([
        trans.Resize((150, 150)),
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5])
    ])

    for i in glob.glob(pred_path):
        pred_dict[i[i.rfind('/*') + 1:]] = prediction(i, transformer, classes)
    path = 'c:/Users/Ram/Desktop/E-complaint-notifier-Django/images\\'
    path += str(imageUploadPage.img1)
    pred.output = pred_dict.get(path)
    if pred.output == "Public Toilets":
        pred.output="PublicToilet"
    pred.fname = "Ram"
    pred.lname = "Gite"

    print(request.GET)
    a = {"output": pred.output,
         "firstname":pred.fname,
         "lastname":pred.lname
         }
    return render(request, 'pred.html', a)


def login(request):
    return render(request, 'upload.html')


def index(request):
    return render(request, 'index.html')

def user_login(request):
    return render(request,'index1.html')


list_details = []
def table_output(request):
    context={}
    list_details.append(".")
    list_details.append(pred.fname)
    list_details.append(pred.lname)
    list_details.append(imageUploadPage.img1)
    list_details.append(pred.output)
    list_details.append("Sandip Foundation Nasik")
    context={
        "data": list_details
    }
    return render(request,'admin.html',context)