import argparse

def parse(args=None, test=None):
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--path', dest='path', type=str, default='../../../Data/unzip_data/Select_jpg/')
    parser.add_argument('--da_ty', dest='da_ty', type=str, default='AP', choices=['AP', 'LL'])
    parser.add_argument('--sam_rate', dest='sam_rate', type=float, default=-1)
    parser.add_argument('--label_name_list', dest='label_name_list', type=str, default='0')

    # network
    parser.add_argument('--network_str', dest='network_str', type=str, default='resnet50',
                        choices=['resnet50', 'densenet121', 'convnext', 'efficientnet', 'vit'])
    parser.add_argument('--mod_str', dest='mod_str', type=str, default='xray',
                        choices=['xray+clin', 'xray', 'clin'])
    parser.add_argument('--pretr_str', dest='pretr_str', type=str, default='nopretr',
                        choices=['nopretr', 'imagenet', 'xray'])
    parser.add_argument('--whe_binary', dest='whe_binary', type=str, default='binary', choices=['binary', 'nobinary'])
    parser.add_argument('--whe_add_loss_weight', dest='whe_add_loss_weight', type=str, default='addlosswei',
                        choices=['addlosswei', 'noaddlosswei'])

    # fusion
    parser.add_argument('--fusion_str', dest='fusion_str', type=str, default='concat', choices=['atten', 'concat'])

    # training
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=90)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--pr_it', dest='pr_it', type=int, default=20)

    return parser.parse_args(args)

def convert(args):
    disli =  ['重症肺炎', '肺炎+呼吸衰竭', '肺炎+低血氧症', '肺炎+胸腔积液', '肺炎+肺不张']
    args.label_name_list = [disli[int(x)] for x in args.label_name_list]

    return args
