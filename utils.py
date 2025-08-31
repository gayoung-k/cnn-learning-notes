import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, scheduler=None, 
                 callbacks=None, device=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Add scheduler
        self.scheduler = scheduler
        self.device = device
        # Add current learning rate variable
        self.current_lr = self.optimizer.param_groups[0]['lr']
        # Receive checkpoint and early stopping classes as a list 
        self.callbacks = callbacks
        
    def train_epoch(self, epoch):
        self.model.train()

        # Calculate running average loss
        accu_loss = 0.0
        running_avg_loss = 0.0
        # Accuracy, total count and accumulated correct count for accuracy calculation
        num_total = 0.0
        accu_num_correct = 0.0
        accuracy = 0.0
        # Visualize real-time training loop progress with tqdm
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1} [Training..]", leave=True) as progress_bar:
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                # Must use to(self.device), not to(device)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # batch 반복 시 마다 누적  loss를 구하고 이를 batch 횟수로 나눠서 running 평균 loss 구함.
                accu_loss += loss.item()
                running_avg_loss = accu_loss /(batch_idx + 1)

                # Calculate accuracy metric
                # Get count of matching predicted class values from outputs and targets
                num_correct = (outputs.argmax(-1) == targets).sum().item()
                # Calculate accuracy using accumulated total count and accumulated correct count for each batch
                num_total += inputs.shape[0]
                accu_num_correct += num_correct
                accuracy = accu_num_correct / num_total

                # Display progress, running average loss and accuracy in tqdm progress_bar
                progress_bar.update(1)
                if batch_idx % 20 == 0 or (batch_idx + 1) == progress_bar.total:  # 20 batch 횟수마다 또는 맨 마지막 batch에서 update
                    progress_bar.set_postfix({"Loss": running_avg_loss,
                                              "Accuracy": accuracy})

        if (self.scheduler is not None) and (not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
            self.scheduler.step()
            self.current_lr = self.scheduler.get_last_lr()[0]
        
        return running_avg_loss, accuracy

    def validate_epoch(self, epoch):
        if not self.val_loader:
            return None

        self.model.eval()

        # Calculate running average loss
        accu_loss = 0
        running_avg_loss = 0
        # Accuracy, total count and accumulated correct count for accuracy calculation
        num_total = 0.0
        accu_num_correct = 0.0
        accuracy = 0.0
        current_lr = self.optimizer.param_groups[0]['lr']
        with tqdm(total=len(self.val_loader), desc=f"Epoch {epoch+1} [Validating]", leave=True) as progress_bar:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)

                    loss = self.loss_fn(outputs, targets)
                    # Calculate accumulated loss for each batch iteration and divide by batch count to get running average loss
                    accu_loss += loss.item()
                    running_avg_loss = accu_loss /(batch_idx + 1)

                    # Calculate accuracy metric
                    # Get count of matching predicted class values from outputs and targets
                    num_correct = (outputs.argmax(-1) == targets).sum().item()
                    # Calculate accuracy using accumulated total count and accumulated correct count for each batch
                    num_total += inputs.shape[0]
                    accu_num_correct += num_correct
                    accuracy = accu_num_correct / num_total
                    
                    # Display progress, running average loss and accuracy in tqdm progress_bar
                    progress_bar.update(1)
                    if batch_idx % 20 == 0 or (batch_idx + 1) == progress_bar.total:  # Update every 20 batches or at the last batch
                        progress_bar.set_postfix({"Loss": running_avg_loss,
                                                  "Accuracy": accuracy})
        # Input the epoch-level calculated loss based on validation data to the scheduler
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(running_avg_loss)
            self.current_lr = self.scheduler.get_last_lr()[0]

        return running_avg_loss, accuracy

    def fit(self, epochs):
        # Create history dict to record training/validation results for each epoch, add learning rate
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate_epoch(epoch)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f} Train Accuracy: {train_acc:.4f}",
                  f", Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}" if val_loss is not None else "",
                  f", Current lr:{self.current_lr:.6f}")
            # Record training/validation results for each epoch, add learning rate
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            history['lr'].append(self.current_lr)

            # If callbacks are provided as constructor arguments, execute below. If early stop should occur, break for loop with is_epoch_loop_break
            if self.callbacks:
                is_epoch_loop_break = self._execute_callbacks(self.callbacks, self.model, epoch, val_loss, val_acc)
                if is_epoch_loop_break:
                    break
                                
        return history

    # Take out each callback from the callbacks list provided as constructor arguments and execute ModelCheckpoint, EarlyStopping
    # Return is_early_stopped to determine early stop when EarlyStopping is called
    def _execute_callbacks(self, callbacks, model, epoch, val_loss, val_acc):
        is_early_stopped = False
        
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if callback.monitor == 'val_loss':    
                    callback.save(model, epoch, val_loss)
                elif callback.monitor == 'val_acc':
                    callback.save(model, epoch, val_acc)
            if isinstance(callback, EarlyStopping):
                if callback.monitor == 'val_loss':
                    is_early_stopped = callback.check_early_stop(val_loss)
                if callback.monitor == 'val_acc':
                    is_early_stopped = callback.check_early_stop(val_acc)
                
        return is_early_stopped

    # Return the trained model
    def get_trained_model(self):
        return self.model
    
class Predictor:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, loader):
        self.model.eval()
        eval_metric = 0.0
        # Total count and accumulated correct count for accuracy calculation
        num_total = 0.0
        accu_num_correct = 0.0

        with tqdm(total=len(loader), desc=f"[Evaluating]", leave=True) as progress_bar:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    pred = self.model(inputs)

                    # Calculate accumulated total count and accumulated total num_correct count for accuracy calculation
                    num_correct = (pred.argmax(-1) == targets).sum().item()
                    num_total += inputs.shape[0]
                    accu_num_correct += num_correct
                    eval_metric = accu_num_correct / num_total

                    progress_bar.update(1)
                    if batch_idx % 20 == 0 or (batch_idx + 1) == progress_bar.total:
                        progress_bar.set_postfix({"Accuracy": eval_metric})
        
        return eval_metric

    def predict_proba(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            # Since we're returning prediction values, targets are not needed
            #targets = targets.to(self.device)
            pred_proba = F.softmax(outputs, dim=-1) #or dim=1

        return pred_proba

    def predict(self, inputs):
        pred_proba = self.predict_proba(inputs)
        pred_class = torch.argmax(pred_proba, dim=-1)

        return pred_class

class EarlyStopping:
    def __init__(self, monitor='val_loss', mode='min', early_patience=5, verbose=1):
        self.monitor = monitor
        self.mode = mode
        self.early_patience = early_patience
        self.verbose = verbose
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0

    def is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value
        else:
            return value > self.best_value

    def check_early_stop(self, value):
        is_early_stopped = False
        
        if self.is_improvement(value):
            self.best_value = value
            self.counter = 0
            is_early_stopped =False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.early_patience}")
            if self.counter >= self.early_patience:
                is_early_stopped = True
                if self.verbose:
                    print("Early stopping happens and train stops")
        
        return is_early_stopped
    
    import os


class ModelCheckpoint:
    def __init__(self, checkpoint_dir='checkpoints', monitor='val_loss', mode='min', save_interval=1, verbose=1):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.verbose = verbose
        self.save_interval = save_interval
        self._make_checkpoint_dir_unless()

    def _make_checkpoint_dir_unless(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
    
    # Check if metric value has improved compared to previous epoch based on mode type and return True/False value
    def is_improvement(self, value):
        if self.mode == 'min':
            return value < self.best_value
        else:
            return value > self.best_value

    # Update self.best_value, only execute when is_improvement() returns True 
    def update_best_value(self, value):
        self.best_value = value

    def save(self, model, epoch, value):
        if self.save_interval == 1:
            if self.is_improvement(value):
                self._checkpoint_save(model, epoch, value)
                self.update_best_value(value)
            
        elif self.save_interval > 1:
            if (epoch + 1) % self.save_interval == 0:
                self._checkpoint_save(model, epoch, value)
                 
        # Don't execute, just for reference (save when model performance improves every save_interval times)
        # if (epoch + 1) % self.save_interval == 0 and self.is_improvement(value):
        #     self.update_best_value(value)
        #     self._checkpoint_save(model, epoch, value)
            
    def _checkpoint_save(self, model, epoch, value):
        checkpoint_path = os.path.join(self.checkpoint_dir, 
                                       f'checkpoint_epoch_{epoch+1}_{self.monitor}_{value:.4f}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        if self.verbose:
            print(f"Saved model checkpoint at {checkpoint_path}")