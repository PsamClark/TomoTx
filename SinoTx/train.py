import torch 
import os
import sys
import torch.nn as nn
import torch.optim as opt
import torch.optim.lr_scheduler as lrs
from torch.nn.parallel import DistributedDataParallel as ddp
from torch.distributed import init_process_group, destroy_process_group

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from data_handler import *

from model import SinoTx

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:

    def __init__(self, model: nn.Module, dl_group,  
                    lr: float = 0.0001, 
                    snapshot_path: str = 'snapshot.pt',
                    dt_string = None, device=None,
                    distributed = True):

        #load dataloaders
        self._trn_dl = dl_group[0]['dl']
        self._vldt_dl = dl_group[2]['dl']
        self._tst_dl = dl_group[1]['dl']

        #load samplers
        self._trn_smplr = dl_group[0]['smplr']
        self._vldt_smplr = dl_group[2]['smplr']
        self._tst_smplr = dl_group[1]['smplr']

        #specify the device configuration 
        self._single_device=False
        if distributed:
            self._device = int(os.environ["LOCAL_RANK"])
            self._global = int(os.environ["RANK"])

        else:
            if device is None:
                self._device = 'cpu'
            else:
                self._device = device

            self._single_device = True

        #send model to device
        self._model = model.to(self._device)

        #init the loss arrays
        self._vldt_losses = []
        self._trn_losses = []

        #set up optimiser
        self._optimiser = opt.Adam(model.parameters(), lr = lr)

        #setup learning rate scheduler 
        self._scheduler =lrs.StepLR(self._optimiser, step_size=20)

        #get datetime information 
        self._dt_string = dt_string 
        self._save_every = 10
        self._snpsht_path = 'models/'+self._dt_string+'_'+ \
            type(self._model).__name__+'_'+snapshot_path

        #init epochs
        self._epochs_run = 0

        #load snapshot if present
        if os.path.exists(snapshot_path):
            print("loading snapshot")
            self._load_snpsht(snapshot_path)

        #distribute model if requested
        if not self._single_device:
            self._model = ddp(self._model,device_ids=[self._device])

    
    def _load_snapshot(self, snapshot_path):

        #indicate location 
        if not self._single_device:
            loc = f"cuda:{self._global}"
        else:            
            loc = self._device

        #load snapshot
        snapshot = torch.load(snapshot_path, map_location=loc)
        self._model.load_state_dict(snapshot["MODEL_STATE"])
        self._epochs_run = snapshot["EPOCHS_RUN"]
        
        print(f"Resuming training from snapshot at Epoch {self._epochs_run}")

    def _save_final_model(self, path):

        #populate model info dict 
        if not self._single_device:

            final = {
                "MODEL_STATE": self._model.module.state_dict(),
                "MODEL": self._model.module(),
            }
        else:    
            final = {
                "MODEL_STATE": self._model.state_dict(),
                "MODEL": self._model,
            }

        #save model 
        torch.save(final, path)

    def _save_snapshot(self, epoch):

        #populate model info dict
        if not self._single_device:

            snapshot = {
                "MODEL_STATE": self._model.module.state_dict(),
                "EPOCHS_RUN": epoch,
            }
        else:    
            snapshot = {
                "MODEL_STATE": self._model.state_dict(),
                "EPOCHS_RUN": epoch,
            }

        #save model
        torch.save(snapshot, self._snpsht_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self._snpsht_path}")

    
    def train(self, epochs:int = 20):
        
        #train across epochs
        for epoch in range(epochs):

            #init epoch loss
            trloss = []
            
            if  not self._single_device:
                self._trn_smplr.set_epoch(epoch)
    
            #train model using dataset
            self._model.train()

            for data in self._trn_dl:

                self._optimiser.zero_grad()

                torch.autograd.set_detect_anomaly(True)

                data = data.to(self._device)

                loss, pred, mask = self._model(data)
                trloss.append(loss.detach().cpu().numpy())

                loss.backward()

                self._optimiser.step()

            #get learning rate and step scheduler
            lr_out = self._optimiser.param_groups[0]["lr"]
            self._scheduler.step()

            #validate the performance of the model
            self._model.eval()
            
            with torch.no_grad():
                vloss = []
                
                if not self._single_device:
                    self._vldt_smplr.set_epoch(epoch)
                    
                for data in self._vldt_dl:

                    data = data.to(self._device)

                    loss, pred, mask = self._model(data)
                    vloss.append(loss.detach().cpu().numpy())


            self._vldt_losses.append(np.mean(vloss))
            self._trn_losses.append(np.mean(trloss))

            #plot losses
            plt.plot(np.arange(len(self._trn_losses)),self._trn_losses,label='train')
            plt.plot(np.arange(len(self._vldt_losses)),self._vldt_losses,label='validate')
            plt.legend()
            plt.savefig('models/'+self._dt_string+'_'+type(self._model).__name__+'.png')
            plt.close()

            if epoch%self._save_every == 0:
                self._save_snapshot(epoch)
                
            #print loss information 
            print(
                f"Epoch : {epoch+1} - loss : {np.mean(trloss):.4f} - val_loss : {np.mean(vloss):.4f} - lr : {lr_out: .4f} \n"
            )

        #test model 
        tloss = []
        self._model.eval()
        
        with torch.no_grad():
            for data in self._tst_dl:

                data = data.to(self._device)
                loss, pred, mask = self._model(data)
                tloss.append(loss.detach().cpu().numpy())

        """
        print(pred[0].shape)
        plt.imshow(pred.cpu()[0])
        plt.savefig('test.png')
        plt.close()
        plt.imshow(mask.cpu()[0][:,None].expand(-1,pred[0].shape[1]))
        plt.savefig('mask.png')
        plt.close()
        """
        print(f"final loss: {np.mean(tloss):.4f}")

        #save model
        self._save_final_model(
                    'models/'+self._dt_string+'_'+type(self._model).__name__+'.pt')
        print('Model saved to:')
        print('models/'+self._dt_string+'_'+type(self._model).__name__+'.pt')

        return pred


def train_main(trainfile: str, vsplit: float, tsplit: float, params,
         snapshot_path: str = "snapshot.pt",
         distributed = True, device=None):

    #set device to cuda if available
    if device is None:
        device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    #create model dir if not already present
    Path("models").mkdir(parents=True, exist_ok=True)

    #create datetime string 
    dt_string=datetime.today().strftime('%y%m%d_%H%M%S')

    #generate model 
    model = generate_model(params)

    #calculate data splits
    tvt_splits = [np.round(1-vsplit-tsplit,1), vsplit, tsplit]
    #if distribution requested 
    if distributed:
        ddp_setup()

    #generate dataloaders 
    dl_group = prepare_training_data(trainfile, tvsplit = tvt_splits,  
                                distrib=distributed)

    #generate trainer object
    trainer = Trainer(model, dl_group=dl_group, snapshot_path=snapshot_path,
                      dt_string=dt_string,distributed = distributed,
                      device=device)

    #train model
    predict = trainer.train(params['train']['maxep'])

    #destroy process group after training
    if distributed:
        destroy_process_group()

    return predict

def generate_model(params):
        
    return SinoTx(seqlen=180, in_dim=256,params=params)


if __name__ == "__main__":
    
    params =read_json(sys.argv[1])
    train_main(params["representation"], params["trn_file"], params["vsplit"],
              params["tsplit"],params["epochs"],params["rand_sample"], params["snpsht_file"])


