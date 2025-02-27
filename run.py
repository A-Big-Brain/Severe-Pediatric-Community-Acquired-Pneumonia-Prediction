import numpy as np
from support_network import Xray_model, get_loss_function, train, test
from torch.utils.data import DataLoader
import torch
import os
from support_dataset import CustomImageDataset, get_transforms
import support_based as spb
import pickle
import pandas as pd

def run_model(args):

    # count classes
    if args.whe_binary == 'binary':
        cla_num = 2
    elif args.whe_binary == 'nobinary':
        cla_num = len(args.label_name_list)

    # create folder
    sa_str = spb.com_mul_str([args.da_ty, args.mod_str, args.sam_rate, args.network_str, args.pretr_str, args.whe_add_loss_weight, args.whe_binary, args.fusion_str, args.batch_size, args.epochs, cla_num])
    sa_fo = spb.path + 'save_results/' + sa_str + '/'
    os.mkdir(sa_fo)

    # create the dataset
    training_data = CustomImageDataset(args.path + args.da_ty + '_tr_label_selected_again.xlsx', args.path + args.da_ty + '/', args.label_name_list,
                                       args.whe_binary, args.sam_rate, get_transforms())
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    test_data = CustomImageDataset(args.path + args.da_ty + '_te_label_selected_again.xlsx', args.path + args.da_ty + '/', args.label_name_list,
                                   args.whe_binary, -1, get_transforms())
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    fea_num = len(training_data.fea_na)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # construct the network
    model = Xray_model(args.network_str, args.pretr_str, cla_num, fea_num, args.mod_str, args.fusion_str)
    model.to(device)

    # loss function
    weight = spb.get_weig(training_data.info, args.label_name_list, args.whe_binary)
    loss_fn = get_loss_function(args.whe_binary, args.whe_add_loss_weight, weight)
    loss_fn.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    # traverse
    res = [[], []]
    for i in range(args.epochs):

        # train
        tr_los = train(model, loss_fn, optimizer, scheduler, train_dataloader, args.pr_it, device)
        print('all_train', i, tr_los)

        # test
        test_loss, pred_li, lab_li, met = test(model, loss_fn, test_dataloader, args.pr_it, device)
        print('all_test', i, test_loss, np.mean(met, 0))

        # add
        res[0].append([tr_los])
        res[1].append([test_loss, pred_li, lab_li, met])

        # save the result
        with open(sa_fo + 'res.txt', 'wb') as fi:
            pickle.dump(res + [args], fi)

        # save the model
        torch.save(model.state_dict(), sa_fo + 'model_weights.pth')

    return

