from transformers import T5Tokenizer, TrainingArguments, Trainer,GPT2TokenizerFast,AutoTokenizer,Adafactor
from torchvision import transforms
import json
import os
import time
from torch.utils.data import DataLoader
import torch
import argparse
from modules.Ped_dataset import PedDataset
from modules.Ped_model import print_trainable_parameters, PedVLMT5
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from tqdm import tqdm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model,epoch, model_name,result_path):
    # Save the model into the designated folder
    path = os.path.join(result_path, timestr, model_name + '_'+ str(epoch)+'.pth')
    torch.save(model, path)


def val_model(dloader, val_model, criterion):
    val_model.eval()
    val_loss = 0

    for idx, (inputs, imgs, labels, i_labels, bodyposes, local_content_features, local_motion_features) in tqdm(enumerate(dloader), total=len(dloader)):
        outputs, out_int = val_model(inputs, imgs, labels, bodyposes, local_content_features, local_motion_features)
        loss_int = criterion(out_int, i_labels)
        vloss = outputs.loss + loss_int
        val_loss += vloss.item()

    return val_loss / len(dloader)


def save_stats(train_loss, val_loss, epochs, lr,result_path):
    stats_dict = {
        'losses': losses,
        'val losses': val_losses,
        'min train loss': train_loss,
        'min val loss': val_loss,
        'epochs': epochs,
        'learning rate': lr,
        'LM': 'T5-Base',
        'Image Embedding': 'Patch'
    }

    # Save stats into checkpoint
    with open(os.path.join(result_path, timestr, 'stats.json'), 'w') as f:
        json.dump(stats_dict, f)


def plot_loss(training_loss, val_loss,result_path):
    num_epochs = len(training_loss)

    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Num epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(result_path, timestr, 'loss.png'))



 

def custom_train(train_loss, val_loss, best_model, epochs, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = Adafactor(model.parameters(),lr=learning_rate,  scale_parameter=True,relative_step=True,warmup_init=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    criterion = torch.nn.CrossEntropyLoss()
    max_grad_norm = 1.0
  

    for epoch in range(epochs, config.epochs):
        print('-------------------- EPOCH ' + str(epoch) + ' ---------------------')
        model.train()
        epoch_loss = 0

        for step, (inputs, imgs, labels, i_lables, bodyposes, local_content_features, local_motion_features) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            # print(inputs.shape, imgs.shape, labels.shape)

            # Forward pass through model
            outputs,out_int = model(inputs, imgs, labels, bodyposes, local_content_features, local_motion_features)
            # print(outputs.logits.shape)
            loss_int = criterion(out_int, i_lables)

            # Calculate loss
         
            loss = (1-config.loss_lambda)*outputs.loss + config.loss_lambda*loss_int
            epoch_loss += loss.item()

            if step % config.checkpoint_frequency == 0:
                print()
                print('Loss: ' + str(loss.item()))

                # Get the hidden states (output)
                hidden_states = outputs.logits

                # Perform decoding (e.g., greedy decoding)
                outputs = torch.argmax(hidden_states, dim=-1)

                text_outputs = [processor.decode(output.to('cpu'), skip_special_tokens=True) for output in outputs]
                text_questions = [processor.decode(q.to('cpu'), skip_special_tokens=True) for q in inputs]
                text_labels = [processor.decode(a.to('cpu'), skip_special_tokens=True) for a in labels]
                print()
                print('Questions:')
                print(text_questions)
                print()
                print('Generated Answers:')
                print(text_outputs)
                print()
                print('Ground Truth Answers:')
                print(text_labels)

            # Back-propogate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Get train and val loss per batch
        epoch_train_loss = epoch_loss / len(train_dataloader)
        losses.append(epoch_train_loss)

        epoch_val_loss = val_model(val_dataloader, model,criterion)
        val_losses.append(epoch_val_loss)

        if not val_loss or min(epoch_val_loss, val_loss) == epoch_val_loss:
            val_loss = epoch_val_loss
        if not train_loss or min(train_loss, epoch_train_loss) == epoch_train_loss:
            train_loss = epoch_train_loss

        # Adjust learning rate scheduler
        scheduler.step()

        print('Training Loss: ' + str(epoch_train_loss))
        print('Validation Loss: ' + str(epoch_val_loss))
        print('---------------------------------------------')

        best_model = deepcopy(model.state_dict())
        # Save model and stats for checkpoints
        save_model(best_model, epoch,'latest_model',config.result_path)
        epochs += 1
        save_stats(train_loss, val_loss, epochs, scheduler.get_last_lr()[0],config.result_path)

    # Save the model and plot the loss
    plot_loss(losses, val_losses,config.result_path)
    return train_loss, val_loss



def save_experiment(statistics):
    """
    Saves the experiment multi_frame_results to a csv
    :param config: The hyperparameters used
    :param statistics: The accuracies for the training, validation, and test sets
    """
    trial_dict = {
        'Model name': [timestr],
        'Learning rate': [config.learning_rate],
        'Weight decay': [config.weight_decay],
        'Batch size': [config.batch_size],
        'Epochs': [config.epochs],
        'LoRA finetuning': [config.lora],
        'GPA Hidden Size': [config.gpa_hidden_size],
        'LoRA Dimension': [config.lora_dim],
        'LoRA Alpha': [config.lora_alpha],
        'LoRA Dropout': [config.lora_dropout],
        'Freeze T5': [config.freeze_lm],
        'Min Training Loss': [statistics[0]],
        'Min Validation Loss': [statistics[1]],
        'Min Testing Loss': [statistics[2]],
    }

    trial_dict = pd.DataFrame(trial_dict)
    trial_dict.to_csv(os.path.join(config.result_path, timestr, 'results.csv'), index=False, header=True)

def params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Model learning rate starting point, default is 1e-4.")
    parser.add_argument("--batch-size", default=1, type=int,
                        help="Batch size per GPU/CPU for training and evaluation, defaults to 4.")
    parser.add_argument("--weight-decay", default=0.05, type=float,
                        help="L2 Regularization, default is 0.05")
    parser.add_argument("--epochs", default=6, type=int,
                        help="Number of epochs to train for, default is 15")
    parser.add_argument('--gpa-hidden-size', default=128, type=int, help='Hidden dimension for Gated Pooling Attention, '
                                                                         'default is 128')
    parser.add_argument('--loss_lambda', default=0.5, type=float, help='Weight for the int loss.')
    parser.add_argument('--freeze-lm',  default=False, help='Freeze LM during training')
    parser.add_argument('--lm', default='T5-Base', choices=['T5-Base', 'T5-Large'], type=str, help='Backbone LM to use, '
                                                                                        'use \'T5-Base\' for T5-Medium')
    parser.add_argument('--checkpoint-frequency', default=500, type=int, help='Frequency of showing example outputs')
    parser.add_argument('--lora', default=False, help='Perform LoRA finetuning, recommend if '
                                                            'using T5-Large backbone LM')
    parser.add_argument('--lora-dim', default=64, type=int, help='LoRA dimension')
    parser.add_argument('--lora-alpha', default=32, type=int, help='LoRA alpha')
    parser.add_argument('--lora-dropout', default=0.05, type=float, help='LoRA dropout')
    parser.add_argument('--num-workers', default=0, type=int, help='# of Workers used by Dataloader')
    parser.add_argument('--optical', default=True,  help='Use optical flow images')
    parser.add_argument('--load-checkpoint', default=False, help='Whether to load a checkpoint from '
                                                                       'multi_frame_results folder')
    parser.add_argument('--checkpoint-file', default='T5-Base', type=str, help='The checkpoint to load from '
                                                                                 'multi_frame_results directory')
    parser.add_argument('--img_path', default='',  help='Path to labels')
    parser.add_argument('--data_path', default=r'C:\Users\DELL\Downloads\Ped_VLM.-main\Ped_Dataset',  help='Path to labels')
    parser.add_argument('--result_path', default=r'C:\Users\DELL\Downloads\Ped_VLM.-main\results',  help='Path to result')
    parser.add_argument('--bodypose_path', default=r'C:\Users\DELL\Downloads\Ped_VLM.-main\data\JAAD\bodypose_yolo',  help='Path to result')
    parser.add_argument('--attention', default=False,  help='Fuse feature using attention')
    parser.add_argument('--num_head', default=8,  help='num_head for attention')
    parser.add_argument('--encoder', default='vit',  help='options, clip,vit,resent50')
    parser.add_argument('--use-bodypose', default=True, type=bool, help='Whether to use body pose data')
    parser.add_argument('--use-local-features', default=True, type=bool, help='Whether to use local content and motion features')
    parser.add_argument('--sequence-length', default=5, type=int, help='Number of frames in temporal sequence for local features')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    timestr = time.strftime("%Y%m%d-%H%M%S")

    config = params()

    losses = []
    val_losses = []
    min_train_loss = None
    min_val_loss = None
    best_model = None
    epochs_ran = 0

    # Load processors and models
    model = PedVLMT5(config)
    model.to(device)
    print('Trainable Parameters for full model')
    print_trainable_parameters(model)

    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base',model_max_length=1024)
        # processor = AutoTokenizer.from_pretrained('/home/farzeen/Downloads/LMTraj-SUP_pretrained_tokenizer/checkpoint/tokenizer/trajectoryspiece-pixel-unigram/',trust_remote_code=False,  use_fast=True)
        processor.add_tokens('<')
    elif config.lm =='GPT':
        processor = GPT2TokenizerFast.from_pretrained('gpt2')
        processor.pad_token = processor.eos_token
    elif config.lm == 'PN':
        processor = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        processor.add_tokens(['<'])
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')
        processor.add_tokens('<')


    train_dset = PedDataset(
        input_file=os.path.join(config.data_path, 'train.json'),
        config=config,
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    val_dset = PedDataset(
        input_file=os.path.join(config.data_path, 'val.json'),
        config=config,
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    test_dset = PedDataset(
        input_file=os.path.join(config.data_path, 'test.json'),
        config=config,
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])
    )
    
    # Set bodypose usage based on config
    train_dset.use_bodypose = config.use_bodypose
    val_dset.use_bodypose = config.use_bodypose
    test_dset.use_bodypose = config.use_bodypose

    # Create Dataloaders
    train_dataloader = DataLoader(train_dset, shuffle=True, batch_size=config.batch_size,
                                  num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    val_dataloader = DataLoader(val_dset, shuffle=True, batch_size=config.batch_size,
                                num_workers=config.num_workers, collate_fn=train_dset.collate_fn)
    test_dataloader = DataLoader(test_dset, shuffle=True, batch_size=config.batch_size,
                                 num_workers=config.num_workers, collate_fn=train_dset.collate_fn)


  

    # Load checkpoint if neccesary:
    if config.load_checkpoint:

        print('Loading model from ' + config.checkpoint_file)

        # Load the model and stats from the checkpoint
        model.load_state_dict(torch.load(os.path.join(config.result_path, config.checkpoint_file,
                                                        'latest_model.pth')))
        best_model = PedVLMT5(config)
        best_model.load_state_dict(torch.load(os.path.join(config.result_path, config.checkpoint_file,
                                                            'latest_model.pth')))

        with open(os.path.join(config.result_path, config.checkpoint_file, 'stats.json'), 'r') as f:
            stats = json.load(f)

        min_train_loss, min_val_loss, losses, val_losses, epochs_ran = stats['min train loss'], stats[
            'min val loss'], stats['losses'], stats['val losses'], stats['epochs']

        print(f'Minimum Training Loss: {min_train_loss}')
        print(f'Training Losses: {losses}')
        print(f'Minimum Validation Loss: {min_val_loss}')
        print(f'Validation Losses: {val_losses}')
        print(f'Epochs ran: {epochs_ran}')
        timestr = config.checkpoint_file
    else:
        checkpoint_path = os.path.join(config.result_path, timestr)
        print(f'All model checkpoints and training stats will be saved in {checkpoint_path}')
        os.mkdir(os.path.join(config.result_path, timestr))

       
        # If loading a checkpoint, use the learning rate from the last epoch
        if config.load_checkpoint:
            lr = stats['learning rate']
        else:
            lr = config.learning_rate

        min_train_loss, min_val_loss = custom_train(min_train_loss, min_val_loss, best_model, epochs_ran, lr)
        # best_model = PedVLMT5(config)
        # best_model.load_state_dict(torch.load(os.path.join(config.result_path, timestr, 'latest_model_2.pth')))
        # best_model.to(device)
        # test_loss = val_model(test_dataloader, best_model)
        # statistics = [min_train_loss, min_val_loss, test_loss]
        # save_experiment(statistics)
