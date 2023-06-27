
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pandas as pd

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import ResNet50_Weights
torch.manual_seed(1)
np.random.seed(1)
import cv2

import numpy as np



class ImageClassification():
    def __init__(self,FC):
        self.FC = FC
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_transforms = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)
        self.setFeatures()
        self.classes= {0:'א', 1:'ב', 2:'ג', 3:'ד', 4:'ה', 5:'ו',
            6:'ח', 7:'ט', 8:'י', 9:'כ',10:'ל', 11:'מ', 12:'נ', 13:'ס',
            14:'ע',15:'פ', 16:'צ', 17:'ק', 18:'ר', 19:'ש',  20:'ת' }

    def setFeatures(self):
        if self.FC==False:
            In_features = self.model.fc.in_features
            self.model.fc = nn.Linear(In_features,21)
            self.model.load_state_dict(torch.load('ISl_big_images_best.pt'))

        else:
            In_features = self.model.fc.in_features
            self.model.fc=nn.Sequential(
                                    nn.Linear(In_features,1050),
                                    nn.Linear(1050,210),
                                    nn.Linear(210,21))
            self.model.load_state_dict(torch.load('ISl_cropped_images_best.pt'))
        self.model.to(self.device)


    def softmax(self,x):
        e_x = torch.exp(x - torch.max(x))  # Subtracting the maximum value for numerical stability
        return e_x / torch.sum(e_x)

    def testletter(self,path):
        test_images = [path]
        test_labels = [-1]
        test_encoded_labels = [-1]

        test_images = pd.Series(test_images, name='file_paths')
        test_labels = pd.Series(test_labels, name='labels')
        test_encoded_labels = pd.Series(test_encoded_labels, name='encoded_labels')
        dfletter = pd.DataFrame(pd.concat([test_images, test_labels, test_encoded_labels], axis=1))

        test_dataset = ASLAlphabet(df=dfletter, transform=self.test_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        ans, pers = self.testModelLetter(test_loader)
        ans = self.classes[ans[0][0]]
        per = pers[0]

        print(ans, round(per, 4))
        return ans

    def testModelLetter(self,test_loader):
        testPred = []
        percents = []
        with torch.no_grad():
            self.model.eval()
            for image, target in test_loader:
                image = image.float()
                image, target = image.to(self.device), target.to(self.device)
                image = image.to(self.device)

                output = self.model(image)
                _, predicted = torch.max(output.data, 1)

                testPred.append(predicted.tolist())
                spot = testPred[-1][0]
                soft = self.softmax(output)
                percent = soft[0][spot]
                percents.append(percent.item())
        return testPred, percents


# Create a dataset from the dataframe
class ASLAlphabet(torch.utils.data.Dataset):
    def __init__(self, df, transform=transforms.Compose([transforms.ToTensor()])):
        self.df = df
        self.transform = transform

    def __len__(self):
        length = len(self.df)
        return length

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 2]
        label = torch.tensor(label)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((224,224))
        img = np.array(image)
        image = self.transform(image=img)["image"]
        return image, label



def main():
    ic = ImageClassification(False)
    vid = cv2.VideoCapture(0)
    while (True):

        # Capture the video frame by frame
        ret, frame = vid.read()
        frame = cv2.flip(frame, 1)
        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('z'):
            return_value, image = vid.read()
            cv2.imwrite('opencv_pic.png', image)
            ic.testletter('opencv_pic.png')
        if cv2.waitKey(1) & 0xFF == ord(' '):
            print()

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()






if __name__ == '__main__':
    main()