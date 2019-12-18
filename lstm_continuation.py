#
#   lstm_continuation
#
#   Simon Verret
#   Reza Nourafkan
#   Andre-Marie Tremablay
#

#%% INITIALIZATION
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
from glob import glob
import json
import argparse
from data_reader import RezaDataset


# PARSING ARGUMENTS AND PARAMETERS FILE
'''
You can use this python script to define hyperparameters in three ways:
    - Through argparse style arguments 
        ex: python lstm_contiuation.py --weight_decay 10
    - Trhough a json file named 'lstm_params.json' or custom name
        ex: python lstm_contiuation.py
        with: 'lstm_params.json' containing
            {
                "weight_decay":10,
                ...
            }
        or: python lstm_contiuation.py --file anyname.json
        with: 'anyname.json' containing
            {
                "weight_decay":10,
                ...
            }
    - By passing the parameters as an object containing the arguments as its members:
        ex: python script.py
        with: 'script.py' containing
            import deepContinuation as dac
            args_dict = {
                "weight_decay": 0,
                ...
            }
            class ObjectView():
                def __init__(self,dict):
                    self.__dict__.update(dict)
            args = ObjectView(args_dict)
            dac.load_data(args)
            ...
    See the the example 'lstm_params.json' for a list of all options, see 'random_search.py' for a script like use.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='lstm_params.json', help='defines default parameters from a .json file')

json_path = parser.parse_known_args()[0].file
if os.path.exists(json_path):
     with open(json_path) as f:
        params = json.load(f)
else:
    print("warning: input file '"+json_path+"' not found") 
    params = {}

def either_json(key, or_default):
    try: return params[key]
    except KeyError: return or_default

# training
parser.add_argument('--samples', type=int, default=either_json('samples',2e6), help='Max number of random samples extracted from the database')
parser.add_argument('--batch_size', type=int, default=either_json('batch_size',200), help='batch size for dataloaders')
parser.add_argument('--epochs', type=int, default=either_json('epochs',200), help='Number of epochs to train.')
# architecture
parser.add_argument('--input_size', type=int, default=either_json('input_size',1), help='Padding size')
parser.add_argument('--hidden_size', type=int, default=either_json('hidden_size',512), help='Size of first hidden layer')
parser.add_argument('--out_size', type=int, default=either_json('out_size',512), help='Size of second hidden layer')
parser.add_argument('--max_grad', type=float, default=either_json('max_grad',5), help='Treshold used to clip gradient and avoid gradient explosion')
# loss
parser.add_argument('--loss', type=str, default=either_json('loss','MSELoss'), help='path to the SigmaRe.csv and Pi.csv files')
# optimization
parser.add_argument('--lr', type=float, default=either_json('lr',0.01), help='Initial learning rate')
parser.add_argument('--warmup', action='store_true', default=either_json('warmup',False), help='linear increase of the learning rate in the first epoch') 
parser.add_argument('--schedule', action='store_true', default=either_json('schedule',False), help='Turn on the learning rate scheduler (plateau)')
parser.add_argument('--factor', type=float, default=either_json('factor',0.5), help='scheduler factor')
parser.add_argument('--patience', type=int, default=either_json('patience',4), help='scheduler plateau size')
# regularization
parser.add_argument('--stop', type=int, default=either_json('stop',16), help='Early stopping limit')
parser.add_argument('--weight_decay', type=float, default=either_json('weight_decay',0), help='L2 regularizer factor of the Adam optimizer')
parser.add_argument('--dropout', type=float, default=either_json('dropout',0), help='Dropout factor on each layer')
# data
parser.add_argument('--path', type=str, default=either_json('path','sdata/'), help='path to the SigmaRe.csv and Pi.csv files')
# hardware
parser.add_argument('--seed', type=int, default=either_json('seed',72), help='Random seed')
parser.add_argument('--num_workers', type=int, default=either_json('num_workers',1), help='number of workers in the dataloaders')
parser.add_argument('--no-cuda', action='store_true', default=either_json('no-cuda',False), help='Disables CUDA')
parser.add_argument('--overwrite', action='store_true', default=either_json('overwrite',False), help='overwrite results file, otherwise appends new results')
parser.add_argument('--save', action='store_true', default=either_json('save',False), help='save pytorch state_dict file')
args = parser.parse_known_args()[0]

def name(args):
    name = 'lstm{}-{}_bs{}_lr{}_wd{}_drop{}{}{}'.format(
                args.input_size,
                args.hidden_size,
                args.batch_size, round(args.lr,3), round(args.weight_decay,3), round(args.dropout,3),
                '_wup' if args.warmup else '',
                '_scheduled{}-{}'.format(round(args.factor,3), round(args.patience,3)) if args.schedule else '')
    return name

def dump_params(args):
    with open("lstm_params_"+name(args)+".json", 'w') as f:
        json.dump(vars(args), f, indent=4)


# FUNCTIONS

def load_data(args):
    print("Loading data")
    dataset = RezaDataset(args.path)

    validation_split = .1
    indices = list(range(len(dataset)))
    split = int(np.floor(validation_split*len(dataset)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=validation_sampler)
    return train_loader,valid_loader

class LSTM_model(nn.Module):    
    def __init__(self, args):
        super(LSTM_model, self).__init__()
        self.step_dim = 1 ## args.input_size would be the sequence lenght, wich is arbitrary for the lstm module
        self.hidden_size = args.hidden_size
        self.out_size = args.out_size
        self.dropout = args.dropout
        self.num_layer = 1

        self.lstm = nn.LSTM(self.step_dim, self.hidden_size, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_size, self.out_size)
        
    def forward(self, inputs, hidden): 
        outputs, hidden = self.lstm(inputs, hidden)
        predictions = self.linear(outputs[-1])
        return predictions.squeeze(1), outputs, hidden
    
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.num_layer, batch_size, self.hidden_size),
                  torch.zeros(self.num_layer, batch_size, self.hidden_size))
        return hidden

def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)

def train(args, device, train_loader, valid_loader): 
    lstm = LSTM_model(args).to(device)
    lstm.apply(init_weights)

    optimizer = torch.optim.Adam(lstm.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    
    if args.loss == "L1Loss":
        criterion = nn.L1Loss()
    elif args.loss == "KLDivLoss":
        criterion = nn.KLDivLoss()
    elif args.loss == "MSELoss":
        criterion = nn.MSELoss()
    else:
        print('WARNING : Unknown loss function required, taking MSELoss instead')
        criterion = nn.MSELoss()
            
    if args.schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    factor=args.factor, patience=args.patience, verbose=True, min_lr=1e-6)

    best_val_loss = 1e6
    last_epoch = 0

    with open('training_'+name(args)+'.csv', 'w' if args.overwrite else 'a') as f:
        f.write('\nepoch')
        f.write('\ttrain_loss')
        f.write('\tval_loss')
        f.write('\tlr\n')

        print('training',name(args))
        for epoch in range(1,args.epochs+1):

            print(' epoch', epoch)
            f.write('{}\t'.format(epoch))
            
            lstm.train()
            avg_train_loss = 0
            train_n_iter = 0

            if epoch==1 and args.warmup:
                print('   linear warm-up of learning rate')
            for batch_number, (inputs, targets)  in enumerate(train_loader):
                # print('batch',batch_number)
                this_batch_size = inputs.shape[0]
                if epoch==1 and args.warmup:
                    tmp_lr = batch_number*args.lr/len(train_loader)
                    for g in optimizer.param_groups:
                        g['lr'] = tmp_lr

                inputs = inputs.to(device).float().view(args.input_size, this_batch_size, 1)
                inputs = torch.flip(inputs, dims=[0])
                hidden = lstm.init_hidden(this_batch_size)
                targets = targets.to(device).float()

                optimizer.zero_grad()
                last_output, all_outputs, hidden = lstm(inputs, hidden)
                loss = criterion(last_output, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(lstm.parameters(), args.max_grad)
                optimizer.step()

                avg_train_loss += loss.item()
                train_n_iter += 1
            avg_train_loss = avg_train_loss/train_n_iter
            print('   average training   loss: {:.9f}'.format(avg_train_loss))
            f.write('{:.9f}\t'.format(avg_train_loss))

            lstm.eval()
            avg_val_loss = 0
            val_n_iter = 0
            for batch_number, (inputs, targets)  in enumerate(valid_loader):
                this_batch_size = inputs.shape[0]
                inputs = inputs.to(device).float().view(args.input_size, this_batch_size, 1)
                inputs = torch.flip(inputs, dims=[0])
                hidden = lstm.init_hidden(this_batch_size)
                targets = targets.to(device).float()
                last_output, all_outputs, hidden = lstm(inputs, hidden)
                loss = criterion(last_output, targets)
                
                avg_val_loss += loss.item()
                val_n_iter += 1
            avg_val_loss = avg_val_loss/val_n_iter
            print('   average validation loss: {:.9f}'.format(avg_val_loss))
            f.write('{:.9f}\t'.format(avg_val_loss))

            f.write('{:.9f}\t'.format(optimizer.param_groups[0]['lr']))
            f.write('\n')
            f.flush()

            if args.save:
                torch.save(lstm.state_dict(), 'state_dict_'+name(args)+'_epoch{}_loss{:.9f}.pt'.format(epoch, avg_val_loss))
            
            if args.schedule:
                scheduler.step(avg_train_loss)
            
            if avg_val_loss < best_val_loss: 
                best_val_loss = avg_val_loss
                best_train_loss = avg_train_loss
                best_epoch = epoch
                early_stop_count = args.stop
                for filename in glob('BEST_loss*_epoch*'+name(args)+'*'):
                    os.remove(filename)
                torch.save(lstm.state_dict(), 'BEST_loss{:.9f}_epoch{}_'.format(avg_val_loss,epoch)+name(args)+'.pt')
            else: 
                early_stop_count -= 1
            if early_stop_count==0:
                print('early stopping limit reached!!')
                break

    results_filename = 'all_bests.csv'
    if not os.path.exists(results_filename):
        with open(results_filename,'w') as f:
            f.write('\t'.join([s for s in [
                        'val_loss',
                        'train_loss',
                        'epoch'
                    ]]))
            f.write('\t')            
            f.write('\t'.join(vars(args).keys()))
            f.write('\n')
    with open(results_filename,'a') as f:
        f.write('\t'.join([str(s) for s in [
                    best_val_loss,
                    best_train_loss,
                    best_epoch
                ]]))
        f.write('\t')
        f.write('\t'.join([str(val) for val in vars(args).values()]))
        f.write('\n')
    
    return lstm

if __name__=="__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda: 
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda") 
        print('using GPU')
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('no GPU available')

    dump_params(args)
    train_loader, valid_loader = load_data(args)
    lstm = train(args, device, train_loader, valid_loader)
    