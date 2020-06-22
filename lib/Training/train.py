import time
import math
import torch
import torch.nn as nn
import copy
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy

from tqdm import tqdm

def train(Dataset, model, criterion, epoch, optimizer, writer, device, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        Dataset (torch.utils.data.Dataset): The dataset
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        writer (tensorboard.SummaryWriter): TensorBoard writer instance
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain print_freq (int) and log_weights (bool).
    """

    # Create instances to accumulate losses etc.
    losses = AverageMeter()
    class_losses = AverageMeter()
    inos_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    in_score = int(args.in_part_score * 10)
    out_score = int(args.out_part_score * 10) +1
    # train
    for i, (pre_inp, pre_target) in enumerate(tqdm(Dataset.train_loader)):
        inp = torch.zeros(pre_inp.size(0), 3, args.patch_size, args.patch_size)
        target = torch.zeros(2, pre_inp.size(0))
        
        # inos_range (0.7 ~ 1.2)
        mm_score = torch.randint(in_score, out_score, size=(pre_inp.size(0),))

        #Cropping 1 <= 0 && 1> resizing
        cur_pos_inp = 0
        for k in range(in_score, out_score):
            index_score = mm_score == k
            if not sum(index_score):
                continue
            temp_inp = pre_inp[index_score, :,:,:]
            temp_target = pre_target[index_score]
            bbox_edge = args.patch_size
            if k*0.1 < 1.0:
                bbox_edge = math.floor(k* 0.1 * args.patch_size)
            
            temp_re_inp =  torch.zeros(temp_target.size(0),3, bbox_edge, bbox_edge)                
            x_start = torch.randint(0, pre_inp.size(2)-bbox_edge, size=(temp_target.size(0),))
            y_start = torch.randint(0, pre_inp.size(2)-bbox_edge, size=(temp_target.size(0),))
            for idx in range(temp_target.size(0)):
                ##original  temp_inp = [B,3,X(256),y(256)]
                ## resize-> temp_inp = [B,3,224,224]
                temp_re_inp[idx] = temp_inp[idx, :,
                     x_start[idx]: x_start[idx]+bbox_edge,
                     y_start[idx]: y_start[idx]+bbox_edge]
            temp_inp = temp_re_inp.to(device)
            del temp_re_inp
            if k*0.1 < 1.0: ## croping -> reize
                temp_inp = torch.nn.functional.interpolate(temp_inp, size=(args.patch_size, args.patch_size), mode='bilinear')
            else: ## cropping -> zero padding
                bbox_edge = math.floor((2.0 - k*0.1) * args.patch_size)
                temp_inp = torch.nn.functional.interpolate(temp_inp, size=(bbox_edge, bbox_edge), mode='bilinear')
                pad_size = math.floor((args.patch_size - bbox_edge)/2)
                counter_pad_size = args.patch_size- bbox_edge - pad_size
                temp_inp = torch.nn.functional.pad(temp_inp,(pad_size,counter_pad_size,pad_size,counter_pad_size))

            inp[cur_pos_inp:cur_pos_inp+temp_inp.size(0)] = temp_inp
            target[0, cur_pos_inp:cur_pos_inp+temp_inp.size(0)] = temp_target
            target[1, cur_pos_inp:cur_pos_inp+temp_inp.size(0)] = torch.ones_like(temp_target)*k*0.1
            cur_pos_inp +=temp_inp.size(0)

            del temp_inp, temp_target
        assert cur_pos_inp == args.batch_size, ("skipping image")

        # measure data loading time
        inp = inp.to(device)
        target = target.to(device)

        class_target = target[0]

        data_time.update(time.time() - end)

        # compute model forward
        output, score = model(inp)

        # calculate loss
        cl, rl = criterion(output, target, score, device, args)
        loss = cl + (args.inos_weight * rl)

        # record precision/accuracy and losses
        prec1 = accuracy(output, class_target)[0]
        prec5 = accuracy(output, class_target, topk=(5,))[0]
        top1.update(prec1.item(), inp.size(0))
        top5.update(prec5.item(), inp.size(0))
        class_losses.update(cl.item(), inp.size(0))
        inos_losses.update(rl.item(), inp.size(0))
        losses.update(loss.item(), inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print progress
        if i % args.print_freq == 0:
            print('Training: [{0}][{1}/{2}]\t' 
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  .format(
                   epoch+1, i, len(Dataset.train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5 = top5))
            if args.debug and i!= 0:
                break

    # TensorBoard summary logging
    writer.add_scalar('train/precision@1', top1.avg, epoch)
    writer.add_scalar('train/precision@5', top5.avg, epoch)
    writer.add_scalar('train/average_loss', losses.avg, epoch)
    writer.add_scalar('train/class_loss',class_losses.avg, epoch)
    writer.add_scalar('train/inos_loss', inos_losses.avg, epoch)

    # If the log weights argument is specified also add parameter and gradient histograms to TensorBoard.
    if args.log_weights:
        # Histograms and distributions of network parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch, bins="auto")
            # second check required for buffers that appear in the parameters dict but don't receive gradients
            if value.requires_grad and value.grad is not None:
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch, bins="auto")

    print(' * Train: Loss {loss.avg:.5f} Prec@1 {top1.avg:.3f}'.format(loss=losses, top1=top1))
