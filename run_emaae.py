"""
Train and evaluate an autoencoder

Author(s): Daniela Wiepert
Last modified: 02/10/2025
"""
#IMPORTS
##built-in
import argparse
import json
import os
from pathlib import Path
import warnings
##third-party
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cottoncandy as cc
import torch
from torch.utils.data import DataLoader
##local
from emaae.io import EMADataset, custom_collatefn
from emaae.models import CNNAutoEncoder
from emaae.loops import train, set_up_train, evaluate

warnings.filterwarnings("ignore", category=UserWarning, module="cottoncandy") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to directory with training data.')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to directory with testing data.')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to directory with validation data.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help="Specify a local directory to save configuration files to. If not saving features to corral, this also specifies local directory to save files to.")
    parser.add_argument("--recursive", action="store_true", 
                        help='Recursively find .wav,.flac,.npz files in the feature and stimulus dirs.')
    parser.add_argument("--encode", action="store_true", 
                        help='Save encodings of test features')
    parser.add_argument("--decode", action="store_true", 
                        help='Save decodings of test features')
    ##cotton candy
    cc_args = parser.add_argument_group('cc', 'cottoncandy related arguments (loading/saving to corral)')
    cc_args.add_argument('--bucket', type=str, default=None,
                         help="Bucket to save extracted features to. If blank, save to local filesystem.")
    ##model specific
    model_args = parser.add_argument_group('model', 'model related arguments')
    model_args.add_argument('--model_config', type=str, default=None, 
                                help='Path to model config json. Default = None')
    model_args.add_argument('--model_type', type=str, default='cnn', 
                                help='Type of autoencoder to initialize.')
    model_args.add_argument('--input_dim', type=int, default=13,
                                help='Input dimensions')
    model_args.add_argument('--inner_size', type=int, default=1024,
                                help='Size of encoder representations to learn.')
    model_args.add_argument('--n_encoder', type=int, default=5, 
                                help='Number of encoder blocks to use in model.')
    model_args.add_argument('--initial_ekernel', type=int, default=5, 
                                help='Size of initial kernel to encoding block.')
    model_args.add_argument('--n_decoder', type=int, default=3, 
                                help='Number of decoder blocks to use in model.')
    model_args.add_argument('--initial_dkernel', type=int, default=5, 
                                help='Size of initial kernel to decoding block.')
    model_args.add_argument('--batchnorm_first', action='store_true',
                                help='Indicate whether to use batchnorm before or after nonlinearity.')
    model_args.add_argument('--exclude_final_norm', action='store_true',
                                help='Indicate whether to exclude final normalization layer before encoding.')
    model_args.add_argument('--final_tanh', action='store_true',
                                help='Indicate whether to use tanh activation as final part of decoder.')
    model_args.add_argument('--checkpoint', type=str, default=None, 
                                help='Checkpoint name of full path to model checkpoint.')
    ##train args
    train_args = parser.add_argument_group('train', 'training arguments')
    train_args.add_argument('--eval_only', action='store_true',
                                help='Specify whether to run only evaluation.')
    train_args.add_argument('--early_stop', action='store_true',
                                help='Specify whether to use early stopping.')
    train_args.add_argument('--patience', type=int, default=15,
                                help='Patience for early stopping.')
    train_args.add_argument('--batch_sz', type=int, default=32,
                                help='Batch size for training.')
    train_args.add_argument('--epochs', type=int, default=500,
                                help='Number of epochs to train for.')
    train_args.add_argument('--lr', type=float, default=1e-3,
                                help='Learning rate.')
    train_args.add_argument('--optimizer', type=str, default='adamw',
                                help='Type of optimizer to use for training.')
    train_args.add_argument('--autoencoder_loss', type=str, default='mse',
                                help='Specify base autoencoder loss type.')
    train_args.add_argument('--sparse_loss', type=str, default='tvl2',
                                help='Specify sparsity loss type [l1, tvl2, filter].')
    train_args.add_argument('--cutoff_freq', type=float, default=0.2,
                            help='Cutoff frequency for low pass filter training')
    train_args.add_argument('--ntaps', type=int, default=51,
                            help='n_taps for firwin filter')
    train_args.add_argument('--weight_penalty', action='store_true',
                                help='Specify whether to add a penalty based on model weights.')
    train_args.add_argument('--alpha', type=float, default=0.25,
                                help='Specify loss weights.')
    train_args.add_argument('--update', action='store_true',
                                help='Specify whether to update alpha.')
    train_args.add_argument('--alpha_epochs', type=int, default=15,
                                help='Specify loss weights.')
    train_args.add_argument('--penalty_scheduler', type=str, default='step',
                                help='Specify what penalty scheduler to use.')
    train_args.add_argument('--lr_scheduler', action='store_true', 
                                help='Specify whether to add an lr scheduler')
    train_args.add_argument('--end_lr', type=float, default=0.0001,
                                help='Specify goal end learning rate.')
    args = parser.parse_args()

    # CONNECT TO CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Model training on GPU')
    else:
        device = torch.device("cpu")

    # PREP DIRECTORIES
    args.train_dir = Path(args.train_dir)
    args.val_dir = Path(args.val_dir)
    args.test_dir = Path(args.test_dir)
    args.out_dir = Path(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    assert args.train_dir.exists() and args.val_dir.exists() and args.test_dir.exists(), 'One of the data directories does not exists'

    # name_str =  f'model_lr{args.lr}e{args.epochs}bs{args.batch_sz}_{args.optimizer}_{args.autoencoder_loss}_{args.sparse_loss}'
    # if args.alpha is not None:
    #     name_str += f'_a{args.alpha}'
    # if args.weight_penalty:
    #     name_str += '_weightpenalty'
    # if args.update:
    #     name_str += f'_{args.penalty_scheduler}'
    # if args.early_stop:
    #     name_str += f'_earlystop'
    # if args.batchnorm_first:
    #     name_str += f'_bnf'
    # if args.final_tanh:
    #     name_str += f'_tanh'
    # save_path = args.out_dir / name_str
    # save_path.mkdir(exist_ok=True)
    #print('Saving results to:', save_path)
    if args.sparse_loss == 'filter':
        assert args.cutoff_freq is not None and args.n_taps is not None
        filter_loss = True
        args.alpha = 1
    else:
        filter_loss = False
    # PREP VARIABLES
    ## SET ALPHA
    if args.update and not args.early_stop:
            args.alpha_epochs = args.epochs #if you're updating but not early stopping, assumes it is updating for the entire epoch, so alpha epochs and epochs should be equivalent

    # LOAD FEATURES
    if args.bucket is not None:
        cci_features = cc.get_interface(args.bucket, verbose=False)
        print("Loading features from bucket", cci_features.bucket_name)
    else:
        cci_features = None
        print('Loading features from local filesystem.')

    # SET UP DATASETS/DATALOADERS
    train_dataset = EMADataset(root_dir=args.train_dir, recursive=args.recursive, cci_features=cci_features)
    val_dataset = EMADataset(root_dir=args.val_dir, recursive=args.recursive, cci_features=cci_features)
    test_dataset = EMADataset(root_dir=args.test_dir, recursive=args.recursive, cci_features=cci_features)

    if not args.eval_only:
        assert not bool(set(train_dataset.files) & set(val_dataset.files)), 'Overlapping files between train and validation set.'
        assert not bool(set(train_dataset.files) & set(test_dataset.files)), 'Overlapping files between train and test set.'
        assert not bool(set(test_dataset.files) & set(val_dataset.files)), 'Overlapping files between val and test set.'
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_sz, shuffle=True, num_workers=4, collate_fn=custom_collatefn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collatefn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collatefn)

    # LOAD/SAVE MODEL CONFIG
    if args.model_config is not None:
        with open(args.model_config, "rb") as f:
            model_config = json.load(f)
    else:
        model_config = {'model_type':args.model_type, 'inner_size':args.inner_size, 'n_encoder':args.n_encoder, 'initial_ekernel':args.initial_ekernel, 'n_decoder':args.n_decoder, 'initial_dkernel':args.initial_dkernel, 'input_dim':args.input_dim, 'checkpoint':args.checkpoint,
                        'epochs':args.epochs, 'learning_rate':args.lr, 'batch_sz': args.batch_sz, 'optimizer':args.optimizer, 'autoencoder_loss':args.autoencoder_loss, 'sparse_loss':args.sparse_loss, 
                        'penalty_scheduler':args.penalty_scheduler, 'weight_penalty':args.weight_penalty, 'alpha': args.alpha, 'alpha_epochs':args.alpha_epochs, 'update':args.update, 'early_stop':args.early_stop, 
                        'patience':args.patience, 'batchnorm_first':args.batchnorm_first, 'final_tanh': args.final_tanh, 'lr_scheduler': args.lr_scheduler, 'end_lr':args.end_lr, 'cutoff_freq':args.cutoff_freq, 'ntaps':args.ntaps}

    if args.eval_only:
        args.lr = model_config['learning_rate']
        args.epochs = model_config['epochs']
        args.batch_sz = model_config['batch_sz']
        args.optimizer = model_config['optimizer']
        args.autoencoder_loss = model_config['autoencoder_loss']
        args.sparse_loss = model_config['sparse_loss']
        args.weight_penalty = model_config['weight_penalty']
        args.alpha = model_config['alpha']
        args.update =  model_config['update']
        args.penalty_scheduler =  model_config['penalty_scheduler']
        args.early_stop = model_config['early_stop']
        args.batchnorm_first = model_config['batchnorm_first']
        args.final_tanh = model_config['final_tanh']
        args.lr_scheduler = model_config['lr_scheduler']
        args.end_lr = model_config['end_lr']
        args.n_encoder = model_config['n_encoder']
        args.n_decoder = model_config['n_decoder']
        args.initial_ekernel = model_config['initial_ekernel']
        args.initial_dkernel = model_config['initial_dkernel']
        args.cutoff_freq = model_config['cutoff_freq']
        args.n_taps = model_config['n_taps']
        ##NTAPS CUTOFF FREQ


    name_str =  f'model_e{args.n_encoder}_iek{args.initial_ekernel}_d{args.n_decoder}_idk{args.initial_dkernel}_lr{args.lr}e{args.epochs}bs{args.batch_sz}_{args.optimizer}_{args.autoencoder_loss}_{args.sparse_loss}'
    if args.sparse_loss == 'filter':
        name_str += f'c{args.cutoff_freq}n{args.n_taps}'
    ### LOSS - CUTOFF FREQ
    if args.alpha is not None:
        name_str += f'_a{args.alpha}'
    if args.weight_penalty:
        name_str += '_weightpenalty'
    if args.update:
        name_str += f'_{args.penalty_scheduler}'
    if args.early_stop:
        name_str += f'_earlystop'
    if args.batchnorm_first:
        name_str += f'_bnf'
    if args.final_tanh:
        name_str += f'_tanh'
    if args.lr_scheduler:
        name_str += f'_explr{args.end_lr}'
    save_path = args.out_dir / name_str
    save_path.mkdir(exist_ok=True)

    if args.model_config is None:
        with open(str(save_path/'model_config.json'), 'w') as f:
            json.dump(model_config,f)

    # INITIALIZE MODEL / LOAD CHECKPOINT IF NECESSARY
    if args.model_type=='cnn':
        model = CNNAutoEncoder(input_dim=model_config['input_dim'], n_encoder=model_config['n_encoder'], n_decoder=model_config['n_decoder'], inner_size=model_config['inner_size'], batchnorm_first=model_config['batchnorm_first'], final_tanh=model_config['final_tanh'], initial_ekernel=model_config['initial_ekernel'], initial_dkernel=model_config['initial_dkernel'])
    else:
        raise NotImplementedError(f'{args.model_type} not implemented.')
    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        #print(checkpoint.keys())
        #print(model.state_dict().keys())
        model.load_state_dict(checkpoint)
    model = model.to(device)


    if not args.eval_only:
        optim, criterion, scheduler = set_up_train(model=model, device =device, optim_type=args.optimizer, lr=args.lr, loss1_type=args.autoencoder_loss,
                                        loss2_type=args.sparse_loss, alpha=args.alpha, weight_penalty=args.weight_penalty,
                                        penalty_scheduler=args.penalty_scheduler, lr_scheduler=args.lr_scheduler, epochs=args.alpha_epochs, end_lr=args.end_lr)

        model = train(train_loader=train_loader, val_loader=val_loader, model=model, 
                      device=device, optim=optim, criterion=criterion, lr_scheduler=scheduler, save_path=save_path, 
                      epochs=args.epochs, alpha_epochs=args.alpha_epochs, update=args.update, 
                      early_stop=args.early_stop, patience=args.patience,weight_penalty=args.weight_penalty,
                      filter_loss=filter_loss, filter_cuttoff=args.cutoff_freq, ntaps=args.ntaps)
        
        #SAVE FINAL TRAINED MODEL
        mpath = save_path / 'models'
        mpath.mkdir(exist_ok=True)
        torch.save(model.state_dict(), str(mpath / f'{model.get_type()}_final.pth'))
    
    #Evaluate

    # if args.eval_only:
    #     lr = model_config['learning_rate']
    #     epochs = model_config['epochs']
    #     batch_sz = model_config['batch_sz']
    #     optimizer = model_config['optimizer']
    #     al = model_config['autoencoder_loss']
    #     sl = model_config['sparse_loss']
    #     name_str =  f'model_lr{lr}e{epochs}bs{batch_sz}_{optimizer}_{al}_{sl}'
    #     alpha = model_config['alpha']
    #     if alpha is not None:
    #         name_str += f'_a{alpha}'
    #     wp = model_config['weight_penalty']
    #     if wp:
    #         name_str += '_weightpenalty'
    #     update =  model_config['update']
    #     if update:
    #         ps =  model_config['penalty_scheduler']
    #         name_str += f'_{ps}'
    #     es = model_config['early_stop']
    #     if es:
    #         name_str += f'_earlystop'
    #     if model_config['batchnorm_firs']':
    #         name_str += f'_bnf'
    #     if model_config['final_tanh']:
    #         name_str += f'_tanh'
    #     save_path = args.out_dir / name_str
    #     save_path.mkdir(exist_ok=True)
       
    print('Saving results to:', save_path)
    metrics = evaluate(test_loader=test_loader, model=model, save_path=save_path, device=device, encode=args.encode, decode=args.decode)
