import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from flask import Flask, request, jsonify

# WHAT TO DO
# 1. 추론과 관련없는 것은 다 날렸습니다.
# 2. 추론을 위한 코드를 추가했습니다.
# 3. 모델을 읽어오기 위한 코드를 추가했습니다. (모델 패쓰 포함)
# 4. Flask를 통한 POST 서비스를 위한 코드를 추가했습니다.

app = Flask(__name__)                   #4
app.debug = True                        #4



# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

@app.route('/recognize', methods=['POST'])          #4
def recog_image():                                  #4
    if 'image' not in request.files:                #4
        return "No image file uploaded", 400        #4
    image_file = request.files['image']             #4
    try:                                            #4
        transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

        model = DigitRecognizer()

        model_path = os.environ.get('MODELPATH')    #3
        if model_path is None:                      #3
            model_path = '../model/'                #3

        model.load_state_dict(torch.load(model_path + 'modelfile')) #3
        model.eval()

        image = Image.open(image_file).convert('L')     #2
        image = transform(image).unsqueeze(0)           #2

        # 추론
        with torch.no_grad():                           #2
            output = model(image)                       #2

        # 결과 출력
        _, predicted = torch.max(output, 1)             #2
        return jsonify({'result':predicted.item()})     #2
    except Exception as e:                              #4
        return f"Error recognizing image: {str(e)}", 500    #4

if __name__ == '__main__':      #4
    app.run()                   #4

""" 

# Define transforms
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

model = DigitRecognizer()

model_path = os.environ.get('MODELPATH')
if model_path is None:
    model_path = '../model/'

model.load_state_dict(torch.load(model_path + 'modelfile'))

# Initialize the model
#model = DigitRecognizer().to(device)


# Evaluation
model.eval()

image_path = "../data/0/img_10701.jpg"

image = Image.open(image_path).convert('L')
image = transform(image).unsqueeze(0)

# 추론
with torch.no_grad():
    output = model(image)

# 결과 출력
_, predicted = torch.max(output, 1)
print('Predicted:', predicted.item())

 """