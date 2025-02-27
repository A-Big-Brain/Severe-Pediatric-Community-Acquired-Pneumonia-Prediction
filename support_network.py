import torch
import torchvision.models as Mod
import torch.nn.functional as F
from torch import nn
import datetime
import numpy as np
import support_vit
from util.pos_embed import interpolate_pos_embed
import support_based as spb

class Xray_model(nn.Module):
    def __init__(self, network_str, pretr_str, cla_num, fea_num, mod_str, fusion_str):
        super().__init__()
        self.mod_str = mod_str
        self.network_str = network_str
        self.pretr_str = pretr_str
        self.fusion_str = fusion_str

        if mod_str == 'xray+clin':
            self.image_extractor, im_fea_num = get_image_fea_extractor(network_str, pretr_str)
            if fusion_str == 'atten':
                self.fusion_model = AttentionFusion(input_dim1=im_fea_num, input_dim2=fea_num, output_dim=im_fea_num + fea_num)
            dim_num = im_fea_num + fea_num
        elif mod_str == 'xray':
            self.image_extractor, im_fea_num = get_image_fea_extractor(network_str, pretr_str)
            dim_num = im_fea_num
        elif mod_str == 'clin':
            self.image_extractor = None
            dim_num = fea_num

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(dim_num, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, cla_num)
        )

    def forward(self, x, fea=None):

        if self.mod_str == 'xray+clin':
            if self.network_str == 'vit' and self.pretr_str == 'xray':
                img_fea = self.image_extractor.forward_features(x)
            else:
                img_fea = self.image_extractor(x)

            if self.fusion_str == 'atten':
                img_fea = self.fusion_model(img_fea, fea)
            elif self.fusion_str == 'concat':
                img_fea = torch.cat((img_fea, fea), dim=1)

        elif self.mod_str == 'xray':
            if self.network_str == 'vit' and self.pretr_str == 'xray':
                img_fea = self.image_extractor.forward_features(x)
            else:
                img_fea = self.image_extractor(x)

        elif self.mod_str == 'clin':
            img_fea = fea

        logits = self.linear_relu_stack(img_fea)

        return logits

# get feature network
def get_image_fea_extractor(network_str, pretr_str):
    if network_str == 'resnet50':
        if pretr_str == 'imagenet':
            model = Mod.resnet50(weights='IMAGENET1K_V1')
        elif pretr_str == 'nopretr':
            model = Mod.resnet50()

        model.fc = nn.Identity()
        fea_dim = 2048

    elif network_str == 'densenet121':
        if pretr_str == 'imagenet':
            model = Mod.densenet121(weights='IMAGENET1K_V1')
        elif pretr_str == 'nopretr':
            model = Mod.densenet121()
        elif pretr_str == 'xray':
            model = Mod.densenet121()
            checkpoint = torch.load('../pretrained_weights/densenet121_CXR_0.3M_mae.pth', map_location='cpu')
            checkpoint_model = checkpoint['model']
            model.load_state_dict(checkpoint_model, strict=False)

        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        fea_dim = 1024

    elif network_str == 'convnext':
        if pretr_str == 'imagenet':
            model = Mod.convnext_small(weights='IMAGENET1K_V1')
        elif pretr_str == 'nopretr':
            model = Mod.convnext_small()

        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        fea_dim = 768

    elif network_str == 'efficientnet':
        if pretr_str == 'imagenet':
            model = Mod.efficientnet_b7(weights='IMAGENET1K_V1')
        elif pretr_str == 'nopretr':
            model = Mod.efficientnet_b7()

        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        fea_dim = 2560

    elif network_str == 'vit':
        if pretr_str == 'imagenet':
            model = Mod.vit_b_16(weights='IMAGENET1K_V1')
            model.heads = nn.Identity()

        elif pretr_str == 'nopretr':
            model = Mod.vit_b_16()
            model.heads = nn.Identity()

        elif pretr_str == 'xray':
            model = support_vit.__dict__['vit_base_patch16'](img_size=224, num_classes=1000, drop_rate=0, drop_path_rate=0.1, global_pool=True)
            checkpoint = torch.load('../pretrained_weights/vit-b_CXR_0.5M_mae.pth', map_location='cpu')
            checkpoint_model = checkpoint['model']
            interpolate_pos_embed(model, checkpoint_model)
            model.load_state_dict(checkpoint_model, strict=False)
        fea_dim = 768

    return model, fea_dim

# attention-based fusion
class AttentionFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(AttentionFusion, self).__init__()

        # Linear layers for transforming the inputs
        self.linear1 = nn.Linear(input_dim1, output_dim)
        self.linear2 = nn.Linear(input_dim2, output_dim)

        # Attention weights
        self.attention_weights = nn.Linear(output_dim, 1)

    def forward(self, modality1, modality2):
        # Transform modalities
        transformed1 = F.relu(self.linear1(modality1))
        transformed2 = F.relu(self.linear2(modality2))

        # Concatenate transformed features
        combined = transformed1 + transformed2  # Element-wise sum

        # Compute attention scores
        attention_scores = self.attention_weights(combined)
        attention_scores = F.softmax(attention_scores, dim=0)

        # Apply attention to the combined features
        attended_features = combined * attention_scores

        return attended_features

# get loss function
def get_loss_function(whe_binary, whe_add_loss_weight, weight):

    if whe_binary == 'binary':
        if whe_add_loss_weight == 'addlosswei':
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        elif whe_add_loss_weight == 'noaddlosswei':
            loss_fn = nn.CrossEntropyLoss()

    elif whe_binary == 'nobinary':
        if whe_add_loss_weight == 'addlosswei':
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        elif whe_add_loss_weight == 'noaddlosswei':
            loss_fn = nn.BCEWithLogitsLoss()

    return loss_fn

# train
def train(model, loss_fn, optimizer, scheduler, dataloader, pr_it, device):
    model.train()
    all_los = 0
    for batch, (img, fea, lab) in enumerate(dataloader):
        img, fea, lab = img.to(device), fea.to(device), lab.to(device)

        log = model(img, fea)

        # loss
        loss = loss_fn(log, lab)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # record
        all_los += loss.item()

        if batch % pr_it == 0:
            print('train', batch, datetime.datetime.now(), loss.item())

    return all_los

# test
def test(model, loss_fn, dataloader, pr_it, device):
    model.eval()
    test_loss, pred_li, lab_li = 0, [], []
    with torch.no_grad():
        for batch, (img, fea, lab) in enumerate(dataloader):
            img, fea, lab = img.to(device), fea.to(device), lab.to(device)

            log = model(img, fea)

            # loss
            loss = loss_fn(log, lab)

            test_loss += loss.item()
            pred = torch.sigmoid(log)
            pred_li.append(pred.cpu())
            lab_li.append(lab.cpu())

            if batch % pr_it == 0:
                print('test', batch, datetime.datetime.now(), loss.item())

    pred_li = torch.concat(pred_li).detach().numpy()
    lab_li = np.concatenate(lab_li, 0)
    met = spb.cal_met(pred_li, lab_li)

    return test_loss, pred_li, lab_li, met