import torch
import time
from datetime import datetime
from utils.metrics import accuracy
from utils.AverageMeter import AverageMeter
from utils.ConfusionMatrix import ConfusionMatrix
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class TwoStreamTrainer(object):
    def __init__(self, simulation) -> None:
        self.sim = simulation
    

    def do(self, mode, model, dl, epoch, criterion, optimizer, writer, log_video=True):
        if mode == 'train':
            train = True
            model.train()
        elif mode == 'test':
            train = False
            model.eval()
        else:
            raise ValueError(f'mode must be either "train" or "test", but was {mode}')

        losses = AverageMeter()
        topk_dict = {f'top{k}': AverageMeter() for k in range(1, 6)}

        confusion_matrix = ConfusionMatrix(n_classes=dl.dataset.num_classes, labels=list(dl.dataset.classes))
        dt = datetime.now()
        print(f'\n[ {dt.strftime("%H:%M | %d-%b")} ] {mode.upper()}, EPOCH {str(epoch).upper()} -----------------')

        start_time = time.time()

        for step, (rgbinput , flowinput, labels) in enumerate(dl):
            rgbinput = rgbinput.cuda()
            flowinput = flowinput.cuda()

            batchsize = labels.size(0)
            labels = labels.cuda()
            classification_outputs = model(rgbinput, flowinput)

            loss = criterion(classification_outputs, labels)
            topk_accuracy = accuracy(classification_outputs.data, labels, topk=range(1, 6))
            losses.update(loss.item(), batchsize)

            # update each element in the topk_dict entry with the current accuracy
            for idx, (k,v) in enumerate(topk_dict.items()):
                v.update(topk_accuracy[idx].item(), batchsize)

            class_preds = classification_outputs.data.argmax(1)
            confusion_matrix.update(class_preds.detach(), labels.detach())

            if train is True:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # do some step logging just to make sure that everything is working as intended
            if step % 100 == 0:
                print(f'{step}/{len(dl)}')
                
            for k, v in topk_dict.items():
                writer.add_scalar(f'{mode}_step/{k}', v.avg, step)

        seconds = time.time() - start_time
        print(f'Epoch {epoch} took {seconds/60:.2f} minutes')

        writer.add_scalar(f'time/{mode}_epoch_seconds', seconds, epoch)
        writer.add_scalar(f'{mode}/loss', losses.avg, epoch)

        for k, v in topk_dict.items():
            writer.add_scalar(f'{mode}/{k}', v.avg, epoch)

        writer.add_image(f'{mode}/confusion_matrix', confusion_matrix.as_img(dpi=500, fontsize=2, label_angle=90, display_values=False), epoch, dataformats='CHW')

        # save data outside of tensorboard 
        self.sim.save_data(topk_dict, subdir='topk', title=f'{mode}_topk_{epoch}')
        self.sim.save_data(confusion_matrix.mat, subdir='confusionmatrix', title=f'{mode}_confusion_{epoch}')

        if epoch == 0:
            self.sim.save_data(dl.dataset.classes, subdir='confusionmatrix', title=f'labels', overwrite=True)


class InferenceTrainer(object):
    def __init__(self) -> None:
        pass
    
    def inference(self, model, dl):
        with torch.no_grad():
            model.eval()
            topk_dict = {f'top{k}': AverageMeter() for k in range(1, 6)}
            dt = datetime.now()
            # print(f'\n[ {dt.strftime("%H:%M | %d-%b")} ]-----------------')


            for step, (inputs, labels) in enumerate(dl):
                # print(f'{step}/ {len(dl)}')

                batchsize = labels.size(0)

                if type(inputs) is list:
                    inputs = [i.cuda() for i in inputs]
                else:
                    inputs = inputs.cuda()
                labels = labels.cuda()

                classification_outputs = model(inputs)

                topk_accuracy = accuracy(classification_outputs.data, labels, topk=range(1, 6))

                # update each element in the topk_dict entry with the current accuracy
                for idx, (k,v) in enumerate(topk_dict.items()):
                    v.update(topk_accuracy[idx].item(), batchsize)

            print(f"Top1 Accuracy is: {topk_dict['top1'].avg:.2f}")
            return topk_dict['top1'].avg
