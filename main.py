import support_args as spa
import run

args = spa.parse()
args = spa.convert(args)
args.sam_rate = -1
args.whe_add_loss_weight = 'addlosswei'
args.mod_str = 'clin'
args.pretr_str = 'imagenet'
args.da_ty = 'LL'
args.network_str = 'densenet121'
run.run_model(args)
