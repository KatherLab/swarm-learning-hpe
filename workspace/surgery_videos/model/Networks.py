import torch
import torch.nn as nn
import torchvision.models.resnet
import timm

class FeatureResNet(nn.Module):
    def __init__(self, num_features = 4096):
        super(FeatureResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.sig = nn.Sigmoid()
        self.feature = nn.Linear(2048, num_features)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.feature(x)

        return self.sig(x)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)


class FeatureAlexNet(nn.Module):
    def __init__(self, num_features=4096):
        super(FeatureAlexNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.sig = nn.Sigmoid()
        self.alexnet.classifier = nn.Sequential(nn.Dropout(),nn.Linear(14080, num_features),
    )

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        x = self.alexnet.classifier(x)

        return self.sig(x)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

class PhaseResNet(nn.Module):
    def __init__(self, num_classes, pretrain = None):
        super(PhaseResNet, self).__init__()
        self.featureNet = FeatureResNet()
        if pretrain is not None:
            self.featureNet.load(pretrain)

        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.featureNet.forward(x)
        x = self.classifier(x)

        return x

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def instantiate(self, model):
        params2 = model.named_parameters()
        params1 = self.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)

class PhaseAlexNet(nn.Module):
    def __init__(self, num_classes, pretrain = None):
        super(PhaseAlexNet, self).__init__()
        self.featureNet = FeatureAlexNet()
        if pretrain is not None:
            self.featureNet.load(pretrain)

        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.featureNet.forward(x)
        x = self.classifier(x)

        return x

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def instantiate(self, model):
        params2 = model.named_parameters()
        params1 = self.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)

class PhaseLSTMResNet(nn.Module):
    def __init__(self, num_classes, lstm_size, pretrain = None):
        super(PhaseLSTMResNet, self).__init__()
        self.featureNet = FeatureResNet()

        if pretrain is not None:
            self.featureNet.load(pretrain)

        self.lstm_size = lstm_size
        self.lstm = nn.LSTM(4096, self.lstm_size, batch_first=True)
        self.classifier = nn.Linear(lstm_size, num_classes)

    def init_hidden(self, batch_size, device=None):
        return (torch.zeros(1, 1, self.lstm_size, device=device),
                torch.zeros(1, 1, self.lstm_size, device=device))

    def forward(self, x, hidden_state=None):
        x = self.featureNet.forward(x)
        #x = x.view(1, x.size(0), -1)
        x = x.unsqueeze(0)
        x, hidden_state = self.lstm(x, hidden_state)
        #x = x.view(x.size(1), -1)
        x = x.squeeze(0)
        x = self.classifier(x)

        return x, hidden_state

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def instantiate(self, model):
        params2 = model.named_parameters()
        params1 = self.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)

class PhaseLSTMConvNext(nn.Module):
    def __init__(self, num_classes, temporal, lstm_size, pretrain = None):
        super(PhaseLSTMConvNext, self).__init__()
        self.model = timm.create_model('convnext_base_384_in22ft1k', pretrained=True)

        self.channels = 1024

        self.temporal = temporal

        if temporal:
            self.lstm_size = lstm_size
            self.lstm = nn.LSTM(self.channels, self.lstm_size, batch_first=True)
            self.classifier = nn.Linear(lstm_size, num_classes)
        else:
            self.classifier = nn.Linear(self.channels, num_classes)

    def init_hidden(self, batch_size, device=None):
        return (torch.zeros(1, 1, self.lstm_size, device=device),
                torch.zeros(1, 1, self.lstm_size, device=device))

    def forward(self, x, hidden_state=None):
        x = self.model.forward_features(x)
        
        x = self.model.head.global_pool(x)
        x = self.model.head.norm(x)
        x = self.model.head.flatten(x)
        #print(x.shape)
        if self.temporal:
            x = x.unsqueeze(0)
            x, hidden_state = self.lstm(x, hidden_state)
            x = x.squeeze(0)
        x = self.classifier(x)

        if self.temporal:
            return x, hidden_state
        
        return x

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def instantiate(self, model):
        params2 = model.named_parameters()
        params1 = self.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)