import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

import src.models as models
from src.utils.imagenet_label import label_dict

PATH = "models/state_dict.pt"
WEIGHTS_URL = "https://download.pytorch.org/models/vgg11-bbd30ac9.pth"


def load_pretrained_weights(model: torch.nn.Module, weights_url: str) -> None:
    state_dict_pretrained = torch.utils.model_zoo.load_url(WEIGHTS_URL,
                                                           progress=True)
    # weights에서 마지막 fc레이어 제거
    layer_names_last = list(model.state_dict().keys())[-2:]
    for layer_name in layer_names_last:
        state_dict_pretrained.pop(layer_name)

    # weights 적용
    state_dict = model.state_dict()
    state_dict_pretrained.update(state_dict)
    model.load_state_dict(state_dict_pretrained)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([.5, .5, .5])
    std = np.array([.5, .5, .5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


def main():
    # 데이터셋 준비
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])

    dataset = torchvision.datasets.ImageFolder(root="data/imgs/train",
                                               transform=transform)
    class_names = dataset.classes

    dataset_sizes = {}
    dataset_sizes["train"] = int(0.8 * len(dataset))
    dataset_sizes["val"] = len(dataset) - dataset_sizes["train"]

    datasets = {}
    datasets["train"], datasets["val"] = torch.utils.data.random_split(
        dataset, [dataset_sizes["train"], dataset_sizes["val"]])

    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x],
                                       batch_size=72,
                                       shuffle=True,
                                       num_workers=8)
        for x in ["train", "val"]
    }

    # 데이터셋 이미지 시각화
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)  # 격자 형태의 이미지
    imshow(out, title=[class_names[x] for x in classes])

    # 모델 생성 및 사전훈련된 가중치 불러오기
    model = models.VGG11(10).cuda()
    # load_pretrained_weights(model, WEIGHTS_URL)

    # 특정 레이어만 학습
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 25
    for epoch in range(num_epochs):  # 데이터셋을 수차례 반복합니다.
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            # if phase == 'train':
            #     model.train()  # 모델을 학습 모드로 설정
            # else:
            #     model.eval()  # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
                inputs, labels = data[0].cuda(), data[1].cuda()

                # 변화도(Gradient) 매개변수를 0으로 만들고
                optim.zero_grad()

                # 순전파 + 역전파 + 최적화를 한 후
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optim.step()

                # 통계를 출력합니다.
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))

    print('Finished Training')


if __name__ == "__main__":
    main()