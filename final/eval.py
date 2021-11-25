import torch
import pandas as pd
import argparse
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def eval():
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = FoodImageFolder(
        './food_data/processed_images/', transform=test_transform, txt_file='./food_data/meta/meta/test.txt')
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, num_workers=2, shuffle=False)

    model = NFNe()
    model = model.cuda()
    model.load_state_dict(torch.load(SAVEPATH+'model_weight.pth'))

    model.eval()
    print('Make an evaluation csv file for kaggle submission...')
    Category = []
    i = 0
    for input, _ in test_loader:
        i+=1
        input = input.cuda()
        output = model(input)
        output = torch.argmax(output, dim=1)
        Category = Category + output.tolist()

    Id = list(range(0, 20665))
    samples = {
       'Id': Id,
       'Category': Category 
    }
    df = pd.DataFrame(samples, columns=['Id', 'Category'])

    df.to_csv(SAVEPATH+'submission.csv', index=False)
    print('Done!!')


if __name__ == "__main__":
    eval()
