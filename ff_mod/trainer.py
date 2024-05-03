
import torch
import numpy as np
from tqdm import tqdm

from ff_mod.networks.abstract_forward_forward import AbstractForwardForwardNet
from ff_mod.callbacks.callback import CallbackList, Callback

from ff_mod.dataloader.factory import DataloaderFactory

class Trainer:
    def __init__(self, unsupervised = False, device = "cuda:0", greedy_goodness = False, is_conv = False, is_cl = False) -> None:
        self.overlay_function = None
        self.__net : AbstractForwardForwardNet = None
        self.device = device
        
        self.callbacks = CallbackList()
        
        self.train_loader = None
        self.test_loader = None
        
        self.unsupervised = unsupervised
        self.fusion_func = None
        
        self.greedy_goodness = greedy_goodness
        self.is_conv = is_conv
        self.is_cl = is_cl
        
        self.subset = None
        self.label_to_index_mapping = None
        self.index_to_label_mapping = None
        
        self.neg_label_to_index_mapping = None
        self.neg_index_to_label_mapping = None
    
    def set_network(self, net : AbstractForwardForwardNet) -> None:
        self.__net = net
        self.overlay_function = net.overlay_function

    def get_network(self) -> AbstractForwardForwardNet:
        return self.__net

    def add_callback(self, callback : Callback):
        self.callbacks.add(callback)
    
    def set_fusion_func(self, fusion_func):
        self.fusion_func = fusion_func
    
    def load_data_loaders(self, in_data = "mnist", batch_size: int = 512, test_batch_size: int = 512, subset = None, **kwargs):
        # Old subset should always be a superset of the subset
        dataloader = DataloaderFactory.get_instance().get(in_data, batch_size=batch_size, **kwargs)
        
        train_loader = dataloader.get_dataloader(split = "train", batch_size = batch_size, subset=subset, **kwargs)
        self.replace_train_loader(train_loader, subset = subset)
        
        self.test_loader = dataloader.get_dataloader(split = "test", batch_size = test_batch_size, subset = subset, **kwargs)

        self.num_classes = DataloaderFactory.get_instance().get_num_classes(in_data)

    def replace_train_loader(self, train_loader, subset = None):
        self.train_loader = train_loader
        
        self.subset = subset
        if subset is None:
            self.subset = np.unique([int(x) for x  in self.train_loader.dataset.targets]).tolist()
            
        self.label_to_index_mapping = {label:index for index, label in enumerate(self.subset)}
        self.index_to_label_mapping = {index:label for index, label in enumerate(self.subset)}

    def get_positive_data(self, x, y):
        if not self.unsupervised and not self.is_conv:
            x_pos = self.overlay_function(x, label = y)
        else:
            x_pos = x.clone().detach()
            
        return x_pos
    
    def get_negative_data(self, x, y):
        
        # The extra should be obtained via the mappings
        class_index = torch.tensor([self.label_to_index_mapping[label.item()] for label in y], device = self.device)
        
        rnd_extra = torch.randint(1, len(self.subset), size = (y.size(0),), device = self.device)
        ind_rnd = (class_index+rnd_extra) % len(self.subset)
        
        y_rnd = torch.tensor([self.index_to_label_mapping[label.item()] for label in ind_rnd], device = self.device)
        
        # Previous method
        #rnd_extra = torch.randint(1, self.num_classes, size = (y.size(0),), device = self.device)
        #y_rnd = (y+rnd_extra)%self.num_classes

        if self.greedy_goodness:
            y_rnd = (y+self.get_best_class(x, y))%self.num_classes
        
        
        if self.is_conv:
            x_neg = x.clone().detach()
        else:
            x_neg = self.overlay_function(x, label = y_rnd)
            
        return x_neg, y_rnd
        
    def train_epoch(self, verbose: int = 1, end_step = None):
        if verbose > 0: print("Train epoch")
            
        self.__net.train()
        for step, (x,y) in  tqdm(enumerate(iter(self.train_loader)), total = len(self.train_loader), leave=True, disable=verbose<2):
            if end_step is not None and step > end_step:
                break
            self.callbacks.on_train_batch_start()

            x, y = x.to(self.device), y.to(self.device)

            # Prepare data
            x_pos = self.get_positive_data(x, y)
            x_neg, y_rnd = self.get_negative_data(x, y)
            
            # Train the network
            self.__net.train_network(x_pos, x_neg, labels = [y,y_rnd])
            
            # Get the predictions
            predicts = self.__net.predict(x, self.num_classes)
            
            self.callbacks.on_train_batch_end(predictions = predicts.cpu(), y = y.cpu())

    @torch.no_grad()
    def get_best_class(self, x, y):
        """ Get the best negative class for each sample in x """
        # TODO Greedy not implemented for subsets 
        alls = torch.zeros((self.num_classes-1, x.size(0)), device = self.device)
        for i in range(1, self.num_classes):
            y_temp = (y+i)%self.num_classes
            
            x_exp = self.__net.get_latent(x, y_temp, 1)
            alls[i-1] = self.__net.layers[0].get_goodness(x_exp)

        return alls.max(0)[1] + 1
    
    @torch.no_grad()
    def test_epoch(self, verbose: int = 1):
        if verbose > 0: print("Test epoch")
        
        accuracy = 0.0
        
        self.__net.eval()
        for step, (x,y) in enumerate(iter(self.test_loader)):
            x, y = x.to(self.device), y.to(self.device)
            
            self.callbacks.on_test_batch_start()
            
            predicts = self.__net.predict(x, self.num_classes)
            accuracy += predicts.eq(y).float().sum().item()
            
            self.callbacks.on_test_batch_end(predictions = predicts.cpu(), y = y.cpu())

        torch.cuda.empty_cache()
        
        return accuracy / len(self.test_loader.dataset)

    def train(self, epochs: int = 2, verbose: int = 1):
        for epoch in range(epochs):
            if verbose > 0: print(f"Epoch {epoch}")
                
            self.train_epoch(verbose=verbose)
            self.callbacks.on_train_epoch_end()
            
            self.test_epoch(verbose=verbose)
            self.callbacks.on_test_epoch_end()
            
            
            self.callbacks.next_epoch()
            self.__net.reset_activity()
            
            torch.cuda.empty_cache()