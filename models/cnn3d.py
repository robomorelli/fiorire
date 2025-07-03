import torch
from torch import nn
from torchmetrics import F1Score, Accuracy, ConfusionMatrix

class CNN3D(nn.Module):
    def __init__(self, cfg, **kwargs): #dilation=1
        super().__init__()

        self.cfg = cfg
        self.dilation = self.cfg.model.dilation


        # Replace all this cfg parameter with kwarg (from tune_parameter of config)
        self.n_classes = self.cfg.model.class_number
        self.in_channel = self.cfg.dataset.in_channel
        self.patch_size = self.cfg.dataset.patch_size #kwargs['num_filter']

        self.act_dict = {'Relu':nn.ReLU, 'Elu':nn.ELU, "Selu":nn.SELU, "LRelu":nn.LeakyReLU}
        self.act = self.act_dict[kwargs['act']]
        self.num_filter = kwargs['num_filter']
        self.filter_size = kwargs['filter_size'] # replace 3 in the old network filter size
        self.num_filter_2 = int(self.num_filter*3/4) + self.num_filter

        dilation = (self.dilation, 1, 1)

        if self.patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, self.num_filter, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, self.num_filter, (self.filter_size, self.filter_size, self.filter_size), stride=(1, 1, 1), dilation=dilation, padding=0
            )
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            self.num_filter, self.num_filter, (self.filter_size, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            self.num_filter, self.num_filter_2, (self.filter_size, self.filter_size, self.filter_size), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            self.num_filter_2, self.num_filter_2, (self.filter_size, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            self.num_filter_2, self.num_filter_2, (self.filter_size, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            self.num_filter_2, self.num_filter_2, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes. It requires crossentropy loss that already include sotmax activation
        # self.act = nn.Sigmoid()
        self.fc = nn.Linear(self.features_size, self.n_classes)

        # weight initialization
        self.apply(self.weight_init)

        # Metrics
        #self.train_acc = Accuracy(num_classes=self.n_classes,average="macro")
        #self.val_acc = Accuracy(num_classes=self.n_classes,average="macro")
        #self.f1_score = F1Score(num_classes=self.n_classes,average="macro")
        #self.train_acc = Accuracy(num_classes=self.n_classes,average="macro")
        self.acc = Accuracy()
        self.f1_score = F1Score()
        self.val_cm = ConfusionMatrix(self.n_classes)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def reset_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.in_channel, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.act()(self.conv1(x))
        x = self.pool1(x)
        x = self.act()(self.conv2(x))
        x = self.pool2(x)
        x = self.act()(self.conv3(x))
        x = self.act()(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x
