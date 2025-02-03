"""
Train and evaluate an autoencoder

Author(s): Daniela Wiepert
Last modified: 01/10/2025
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
    parser.add_argument("--debug", action="store_true")
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
    model_args.add_argument('--inner_size', type=int, default=1024,
                                help='Size of encoder representations to learn.')
    model_args.add_argument('--n_encoder', type=int, default=5, 
                                help='Number of encoder blocks to use in model.')
    model_args.add_argument('--n_decoder', type=int, default=3, 
                                help='Number of decoder blocks to use in model.')
    model_args.add_argument('--checkpoint', type=str, default=None, 
                                help='Checkpoint name of full path to model checkpoint.')
    ##train args
    train_args = parser.add_argument_group('train', 'training arguments')
    train_args.add_argument('--eval_only', action='store_true',
                                help='Specify whether to run only evaluation.')
    train_args.add_argument('--batch_sz', type=int, default=2,
                                help='Batch size for training.')
    train_args.add_argument('--epochs', type=int, default=2,
                                help='Number of epochs to train for.')
    train_args.add_argument('--lr', type=float, default=3e-4,
                                help='Learning rate.')
    train_args.add_argument('--optimizer', type=str, default='adamw',
                                help='Type of optimizer to use for training.')
    train_args.add_argument('--autoencoder_loss', type=str, default='mse',
                                help='Specify base autoencoder loss type.')
    train_args.add_argument('--sparse_loss', type=str, default='l1',
                                help='Specify sparsity loss type.')
    train_args.add_argument('--weight_penalty', action='store_true',
                                help='Specify whether to add a penalty based on model weights.')
    train_args.add_argument('--penalty_scheduler', type=str, default='step',
                                help='Specify what penalty scheduler to use.')
    train_args.add_argument('--alpha', type=float, default=None,
                                help='Specify loss weights.')
    args = parser.parse_args()

    #
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Model training on GPU')
    else:
        device = torch.device("cpu")

    #Prepare directories
    args.train_dir = Path(args.train_dir)
    args.val_dir = Path(args.val_dir)
    args.test_dir = Path(args.test_dir)
    args.out_dir = Path(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    assert args.train_dir.exists() and args.val_dir.exists() and args.test_dir.exists(), 'One of the data directories does not exists'
    
    #Load features
    if args.bucket is not None:
        cci_features = cc.get_interface(args.bucket, verbose=False)
        print("Loading features from bucket", cci_features.bucket_name)
    else:
        cci_features = None
        print('Loading features from local filesystem.')

    train_dataset = EMADataset(root_dir=args.train_dir, recursive=args.recursive, cci_features=cci_features)
    val_dataset = EMADataset(root_dir=args.val_dir, recursive=args.recursive, cci_features=cci_features)
    test_dataset = EMADataset(root_dir=args.test_dir, recursive=args.recursive, cci_features=cci_features)

    #DEBUG:
    assert not bool(set(train_dataset.files) & set(val_dataset.files)), 'Overlapping files between train and validation set.'
    assert not bool(set(train_dataset.files) & set(test_dataset.files)), 'Overlapping files between train and test set.'
    assert not bool(set(test_dataset.files) & set(val_dataset.files)), 'Overlapping files between val and test set.'
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_sz, shuffle=True, num_workers=4, collate_fn=custom_collatefn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collatefn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collatefn)

    #LOAD/SAVE MODEL CONFIG
    if args.model_config is not None:
        with open(args.model_config, "rb") as f:
            model_config = json.load(f)
    else:
        model_config = {'inner_size':args.inner_size, 'n_encoder':args.n_encoder, 'n_decoder':args.n_decoder, 'input_dim':13, 'model_type':args.model_type, 'checkpoint':args.checkpoint,
                        'epochs':args.epochs, 'learning_rate':args.lr, 'optimizer':args.optimizer, 'loss1':args.autoencoder_loss, 'loss2':args.sparse_loss, 
                        'penalty_scheduler':args.penalty_scheduler, 'weight_penalty':args.weight_penalty, 'alpha': args.alpha}
    
    lr = model_config['learning_rate']
    e = model_config['epochs']
    save_path = args.out_dir / f'model_lf{lr}_epochs{e}_{args.optimizer}_{args.autoencoder_loss}_{args.sparse_loss}_{args.penalty_scheduler}_weightpenalty{int(args.weight_penalty)}'
    os.makedirs(save_path, exist_ok=True)
    print('Saving results to:', save_path)

    if args.model_config is None:
        with open(str(save_path/'model_config.json'), 'w') as f:
            json.dump(model_config,f)

    #Initialize model
    if args.model_type=='cnn':
        if args.checkpoint is None:
            model = CNNAutoEncoder(input_dim=model_config['input_dim'], n_encoder=model_config['n_encoder'], n_decoder=model_config['n_decoder'], inner_size=model_config['inner_size'])
        else:
            model = torch.load(args.checkpoint)
        model = model.to(device)
    else:
        raise NotImplementedError(f'{args.model_type} not implemented.')

    #Train
    if args.alpha is None:
        args.alpha = 0.25
        update=True
    else:
        update=False

    if not args.eval_only:
        optim, criterion = set_up_train(model=model,optim_type=args.optimizer, lr=args.lr, loss1_type=args.autoencoder_loss, loss2_type=args.sparse_loss, 
                     penalty_scheduler=args.penalty_scheduler, alpha=args.alpha, epochs=args.epochs, weight_penalty=args.weight_penalty)
        
        model = train(train_loader=train_loader, val_loader=val_loader, model=model, optim=optim, criterion=criterion, 
                      epochs=args.epochs, save_path=save_path, device=device, weight_penalty=args.weight_penalty,update=update,debug=args.debug)

        #SAVE FINAL TRAINED MODEL
        torch.save(model, str(save_path / f'{model.get_type()}.pth'))
    
    #Evaluate
    metrics = evaluate(test_loader=test_loader, model=model, save_path=save_path, device=device)
