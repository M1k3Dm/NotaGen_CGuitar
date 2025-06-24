import os
import time
import json
import wandb
import torch
import random
import numpy as np
from abctoolkit.transpose import Key2index, Key2Mode
from utils import *
from config import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GPT2Config, get_constant_schedule_with_warmup

def load_model(device: torch.device) -> NotaGenLMHeadModel:
    
    print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model
 
def initialize_training(device):

    patch_cfg = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS,
        max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH,
        n_embd=HIDDEN_SIZE,
        num_attention_heads=HIDDEN_SIZE // 64,
        vocab_size=1,
    )

    byte_cfg = GPT2Config(
        num_hidden_layers=CHAR_NUM_LAYERS,
        max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1,
        n_embd=HIDDEN_SIZE,
        num_attention_heads=HIDDEN_SIZE // 64,
        vocab_size=128,
    )

    model = NotaGenLMHeadModel(encoder_config=patch_cfg, decoder_config=byte_cfg)
    # Freeze patch level decoder parameters
    for p in model.patch_level_decoder.parameters():
        p.requires_grad = False 
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INIT] Model instantiated: total params={total:,}, trainable={trainable:,}")
    
    model.to(device)
    print(f"[INIT] Model moved to {device}")
    
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY)
    print(f"[INIT] Optimizer: AdamW(lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")

    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer, 
        num_warmup_steps=WARMUP_STEPS)
    
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True)

    print(f"[INIT] LR schedulers: warmup_steps={WARMUP_STEPS}, plateau(patience=2)")
    return model, scaler, optimizer, lr_scheduler, plateau_scheduler

class NotaGenDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        base_path  = self.filenames[idx]['path'] 

        with open(base_path, "r", encoding="utf-8") as f:
            abc_text = f.read()

        file_bytes = patchilizer.encode_train(abc_text)
        file_masks = [1] * len(file_bytes)

        file_bytes = torch.tensor(file_bytes, dtype=torch.long)
        file_masks = torch.tensor(file_masks, dtype=torch.long)
        return file_bytes, file_masks

def collate_batch(input_batches):
    
    input_patches, input_masks = zip(*input_batches)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)

    return input_patches, input_masks 
  
# Do one epoch for training
def train_epoch(epoch):
    
    # Wrap the training dataset in a progress bar
    tqdm_train_set = tqdm(train_loader)
    total_train_loss = 0.0

    # Set model to training mode 
    model.train()

    # Iterate over batches, counting from 1 for averaging
    for iter_idx, (batch_patches, batch_masks) in enumerate(tqdm_train_set, start=1):

        # Move inputs to the target device (GPU/CPU)
        batch_patches = batch_patches.to(device, non_blocking=True)
        batch_masks = batch_masks.to(device, non_blocking=True)
        
        # Mixed-precision forward pass
        with autocast():
            output = model(batch_patches, batch_masks)
            loss = output.loss

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()     

        # Unscale grads before clipping to avoid Inf/NaN
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        # Perform optimizer step and update the scaler
        scaler.step(optimizer)
        scaler.update()

        # Step the learning-rate scheduler (per batch)
        lr_scheduler.step()

        # Clear gradients for the next step
        model.zero_grad(set_to_none=True)

        # Accumulate running loss and compute average
        total_train_loss += loss.item()
        avg_train_loss = total_train_loss / iter_idx

        # Update tqdm display with current average loss
        tqdm_train_set.set_postfix({'Train Loss': avg_train_loss})

    # Return the epoch's average training loss
    return total_train_loss / iter_idx   

# do one epoch for eval
def eval_epoch():

    # Wrap the evaluation dataset in a progress bar
    tqdm_eval_set = tqdm(eval_loader)
    total_eval_loss = 0.0

    # Set model to evaluation mode 
    model.eval()

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate over evaluation batches
        for iter_idx, (batch_patches, batch_masks) in enumerate(tqdm_eval_set, start=1):
            
            # Move inputs to the target device
            batch_patches = batch_patches.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)

            # Forward pass to compute loss
            output = model(batch_patches, batch_masks)
            loss = output.loss

            # Accumulate loss and compute running average
            total_eval_loss += loss.item()
            avg_eval_loss = total_eval_loss / iter_idx

            # Update progress bar display with current average loss
            tqdm_eval_set.set_postfix({'Eval Loss': avg_eval_loss})

    # After all batches, step the plateau LR scheduler with the final average loss
    plateau_scheduler.step(avg_eval_loss)    

    # Return the epoch's average evaluation loss
    return avg_eval_loss

# train and eval
if __name__ == "__main__":
        
    # Initialize wandb
    if WANDB_LOGGING:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="NG_CGuitar",
                   name=WANDB_NAME)
        wandb.define_metric("epoch")
        wandb.define_metric("Loss/Train", step_metric="epoch")
        wandb.define_metric("Loss/Eval", step_metric="epoch")
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
   
    patchilizer = Patchilizer()

    # Load and shuffle data
    with open(DATA_TRAIN_INDEX_PATH, "r", encoding="utf-8") as f:
        print("Loading Train Data...")
        train_files = [json.loads(line) for line in f]
    
    with open(DATA_EVAL_INDEX_PATH, "r", encoding="utf-8") as f:
        print("Loading Eval Data...")
        eval_files = [json.loads(line) for line in f]

    random.shuffle(train_files)
    random.shuffle(eval_files)

    # Create a Datasets for training and evaluation examples
    train_set = NotaGenDataset(train_files)
    eval_set = NotaGenDataset(eval_files)

    # Wrap the Datasets in DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

    # Device: GPU if available, else CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Initialize model, AMP scaler, optimizer and LR schedulers on chosen device
    model, scaler, optimizer, lr_scheduler, plateau_scheduler = initialize_training(device)

    # Decide whether to start from a pretrained model or resume from a training checkpoint
    if not LOAD_FROM_CHECKPOINT:
        if os.path.exists(PRETRAINED_PATH):
            # Load pre-trained checkpoint to device
            checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
            model.load_state_dict(checkpoint['model'])
            print(f"[INIT] Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['min_eval_loss']}")
            
            # Initialize tracking variables for a fresh run
            pre_epoch = 0
            best_epoch = 0
            min_eval_loss = float('inf')
        else:
            raise Exception('Pre-trained Checkpoint not found. Please check your pre-trained ckpt path.')
            
    else:
        if os.path.exists(WEIGHTS_PATH):
            # Load checkpoint to device
            checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
            model.load_state_dict(checkpoint['model'])

            # Restore optimizer and scheduler state
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_sched'])
            plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler'])

            # Restore variables
            pre_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            min_eval_loss = checkpoint['min_eval_loss']
            print("[INIT] Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
            checkpoint = None
        else:
            raise Exception('Checkpoint not found to continue training. Please check your parameter settings.')
    
    epochs_without_improve = 0
    for epoch in range(1+pre_epoch, NUM_EPOCHS+1):
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        
        # Run one epoch of training and evaluation
        train_loss = train_epoch(epoch)
        eval_loss = eval_epoch()

        # Append epoch losses + timestamp to log file
        with open(LOGS_PATH,'a') as f:
            f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")

        if WANDB_LOGGING:
            log_dict = {
                "Loss/Train": train_loss,
                "Loss/Eval": eval_loss,
            }
            wandb.log(log_dict)

        # Check for improvement and save best model   
        if eval_loss < min_eval_loss:
            best_epoch = epoch
            min_eval_loss = eval_loss
            epochs_without_improve = 0
            checkpoint = { 
                    'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict(),
                    'plateau_scheduler': plateau_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'min_eval_loss': min_eval_loss
            }
            torch.save(checkpoint, WEIGHTS_PATH)
        else:
            epochs_without_improve += 1

        # Early stopping
        if epochs_without_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs).")
            with open(LOGS_PATH,'a') as f:
                f.write(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs).\n")
            break 

    print("Best Eval Epoch : "+str(best_epoch))
    print("Min Eval Loss : "+str(min_eval_loss))

    if WANDB_LOGGING:
        wandb.finish()
        
    with open(LOGS_PATH, "a") as f:
        f.write(f"Best Eval Epoch : {best_epoch}\n"
                f"Min Eval Loss  : {min_eval_loss}\n")    