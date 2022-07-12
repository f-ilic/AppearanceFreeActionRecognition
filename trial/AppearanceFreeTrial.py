from matplotlib import pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from pytorchvideo.data.encoded_video import EncodedVideo

from dataset.db_factory import DBfactory
from config import cfg, build_cfg
import time
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Button, Slider
import matplotlib as mpl
import random
import pandas as pd
from datetime import datetime
from os.path import join
import os
from trial.plot_scripts.averaged_results import main as save_visualisation_plots
from trial.plot_scripts.averaged_results import main_from_csv as visualize_plots
import time
import numpy as np
from trial.plot_scripts.util import collect_all_user_data, get_user_csv

mpl.rcParams['toolbar'] = 'None'

cfg = build_cfg()
class Logger(object):
    def __init__(self, userID, savedir) -> None:
        super().__init__()
        self.userID = userID
        self.df = pd.DataFrame()
        self.savedir = savedir

    def append(self, data):
        self.df = self.df.append(data, ignore_index=True)
        self.df = self.df.convert_dtypes()


    def save(self):
        outpath = self.savedir
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        # print(self.df)
        outfile = join(outpath,self.userID)
        # print(f"probably should be saving to {outfile}")
        savepath = join(outpath,f'{self.userID}.csv')
        self.df.to_csv(savepath)

class AppearanceFreeTrial(object):
    def __init__(self, vol, lbl, ucf101_labels, display_txt, trainmode):
        # vol has to be shape C,T, H, W
        self.vol = vol.clip(0,1)
        self.ucf101_labels = ucf101_labels.copy()
        self.slices = vol.shape[1]
        self.ind = 0
        self.fig, self.ax = plt.subplots()
        self.lbl = lbl
        self.display_txt = display_txt
        self.im = self.ax.imshow(self.vol[:, self.ind, ...].permute(1,2,0))
        plt.subplots_adjust(left=0.28)
        self.ax.axis('off')
        rax = plt.axes([0, 0.3, 0.35, 0.55], frameon=False ,aspect='equal')
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

        # remove the one label it actually is from the list, so we cant get doubles in here
        self.ucf101_labels.remove(lbl)
        labels = [f'{i}' for i in random.sample(self.ucf101_labels, k=4)]
        labels.append(f'{lbl}')
        labels.sort()



        axnext = plt.axes([0.1, 0.3, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next)

        plt.text(-0.25, -1, self.display_txt)
        self.update()
        plt.show(block=False)
        self.next_clicked = False

        axfps = plt.axes([0.45, 0.05, 0.35, 0.03])
        self.fps_slider = Slider(
            ax=axfps,
            label='Framerate',
            valmin=10,
            valmax=40,
            valinit=25,
            )

        labels_appended = []
        if trainmode:
            self.ax.set_title(f'The correct Label is: {self.lbl}')
            plt.text(-27,25, "\nWARM-UP!", fontdict={'family' : 'DejaVu Sans','weight' : 'bold','size'   : 20})
            
            for l in labels:
                if self.lbl == l:
                    labels_appended.append(l + " <-------")
                else:
                    labels_appended.append(l)
        else:
            labels_appended = labels

        self.radios = RadioButtons(rax, labels_appended)
        for circle in self.radios.circles: # adjust radius here. The default is 0.05
            circle.set_radius(0.04)

        self.start()

    def next(self, event):
        self.next_clicked = True
        plt.close()

        res = {
            "GroundTruth": self.lbl,
            "Prediction": self.radios.value_selected,
            "Correct": int(self.radios.value_selected == self.lbl),
            "Fps": self.fps_slider.val,
            "NumberOfFrames": self.slices
        }
        self.res = res

    def handle_close(self, event):
        self.next_clicked = True

    
    def start(self):
        fig = plt.gcf()
        fig.canvas.set_window_title(WINDOWTITLE)
        while not self.next_clicked:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            start_time = time.time()
            self.ind = (self.ind + 1) % self.slices
            self.update()
            elapsed_time = time.time() - start_time

            if elapsed_time > 1/self.fps_slider.val:
                pass
            else:
                plt.pause((1/self.fps_slider.val) - elapsed_time)

    def update(self):
        self.im.set_data(self.vol[:, self.ind, :, :].permute(1,2,0))
        self.im.axes.figure.canvas.draw()



class MessageScreen(object):
    def __init__(self, s, button_text = "Continue") -> None:
        super().__init__()
        fig = plt.gcf()
        fig.canvas.set_window_title(WINDOWTITLE)

        # axnext =  plt.axes([0.1, 0.2, 0.15, 0.075])
        axnext =  plt.axes([0.75, 0.10, 0.15, 0.075])

        bnstart = Button(axnext, button_text)
        bnstart.on_clicked(self.start)
        plt.text(-4.5, 3, s)
        plt.show(block=True)


    def start(self, event):
        plt.close()

class DemoActionsScreen(object):
    def __init__(self, button_text = "Continue", modality='ucf5') -> None:
        super().__init__()

        if modality == 'ucf5':
            simple_txt = 'These are "Regular Videos"'
        if modality == 'afd5':
            simple_txt = 'These are "Appearance Free" videos'

        self.button_text = button_text
        self.fig, self.ax = plt.subplots(2,3, figsize=(12,6))
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        self.fig.suptitle(f"Get familiar with the following actions!\n{simple_txt}")


        axes = list(self.ax.ravel())
        for a in axes:
            a.axis('off')
        del axes[-1]

        self.classes = ['JumpingJack', 'Lunges', 'PullUps', 'PushUps', 'Swing']
        self.ax_obj = {k:v for k, v in zip(self.classes, axes)}

        video_paths = [ f'/home/f/projects/MotionPaper/data/{modality}/JumpingJack/v_JumpingJack_g18_c02.avi',
                        f'/home/f/projects/MotionPaper/data/{modality}/Lunges/v_Lunges_g24_c01.avi',
                        f'/home/f/projects/MotionPaper/data/{modality}/PullUps/v_PullUps_g19_c01.avi',
                        # f'/home/f/projects/MotionPaper/data/{modality}/PullUps/v_PullUps_g21_c02.avi',
                        f'/home/f/projects/MotionPaper/data/{modality}/PushUps/v_PushUps_g05_c04.avi',
                        f'/home/f/projects/MotionPaper/data/{modality}/Swing/v_Swing_g08_c03.avi']

        self.curr_vid_idx = {'JumpingJack': 0, 'Lunges':0, 'PullUps':0, 'PushUps':0, 'Swing':0}
        self.curr_vid_max_idx = {}
        self.im_axes = {}
        self.vols = {}

        for curr_axis, curr_path, curr_class in zip(axes, video_paths, self.classes):
            video = EncodedVideo.from_path(curr_path)
            start_sec = 0 
           
            end_sec = video.duration
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)['video']
            vol_to_append = video_data.permute(1,2,3,0)/255.
            self.vols[curr_class] = vol_to_append

            first_frame = np.zeros_like(vol_to_append[0].numpy())
            self.im_axes[curr_class] = curr_axis.imshow(first_frame)
            curr_axis.set_title(curr_class)
            self.curr_vid_max_idx[curr_class] = int(video_data.shape[1])

        self.tmp_classes = self.classes.copy()
        self.playing_classes = [self.tmp_classes.pop(0)]
        self.update()
        plt.show(block=False)
        self.next_clicked = False

        self.fig.canvas.set_window_title(WINDOWTITLE)
        axnext =  plt.axes([0.75, 0.10, 0.15, 0.075])
        self.bnext = Button(axnext, self.button_text)
        self.bnext.on_clicked(self.next)

        self.bnext.ax.patch.set_visible(False)
        self.bnext.label.set_visible(False)
        self.bnext.ax.axis('off')
        plt.gcf().canvas.draw()

        self.start()


    def start(self):
        play_time = time.time()
        fps = 25
        while not self.next_clicked:
            start_time = time.time()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            for cls in self.playing_classes:    
                self.curr_vid_idx[cls] = (self.curr_vid_idx[cls] + 1) % self.curr_vid_max_idx[cls]
                self.update()
            
            elapsed_time = time.time() - start_time

            if elapsed_time > 1/fps:
                pass
            else:
                plt.pause((1/fps) - elapsed_time)

            if time.time()-play_time > 2 and len(self.tmp_classes)>0:
                self.playing_classes.append(self.tmp_classes.pop(0))
                play_time=time.time()

            if len(self.tmp_classes) == 0:
                self.bnext.ax.patch.set_visible(True)
                self.bnext.label.set_visible(True)
                self.bnext.ax.axis('on')



    def next(self, event):
        self.next_clicked = True
        plt.close()

    
    def handle_close(self, event):
        self.next_clicked = True

    def update(self):
        for curr_class in self.playing_classes:
            i = self.curr_vid_idx[curr_class]
            self.im_axes[curr_class].set_data(self.vols[curr_class][i,...])


class CompositeScreen(object):
    def __init__(self):


        curr_path = 'trial/composite_demo.mp4'

        video = EncodedVideo.from_path(curr_path)
        start_sec = 0 

        end_sec = video.duration
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)['video']
        self.vol = video_data/255.
        
        # vol has to be shape C,T, H, W
        self.vol = self.vol.clip(0,1)
        self.slices = self.vol.shape[1]
        self.ind = 0
        self.fig, self.ax = plt.subplots()
    
        self.im = self.ax.imshow(self.vol[:, self.ind, ...].permute(1,2,0))
        self.ax.axis('off')
        self.ax.set_aspect(1)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        plt.subplots_adjust(bottom=0.3)

      
        axnext =  plt.axes([0.75, 0.10, 0.15, 0.075])
        bnext = Button(axnext, 'Continue')
        bnext.on_clicked(self.next)
        self.ax.set_title("Example that shows how the modalities are connected")
        self.update()
        self.next_clicked = False
        # plt.tight_layout()
        plt.show(block=False)

        self.start()

    def next(self, event):
        self.next_clicked = True
        plt.close()

    def handle_close(self, event):
        self.next_clicked = True
    
    def start(self):
        fig = plt.gcf()
        fig.canvas.set_window_title(WINDOWTITLE)
        while not self.next_clicked:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            start_time = time.time()
            self.ind = (self.ind + 1) % self.slices
            self.update()
            elapsed_time = time.time() - start_time

            if elapsed_time > 1/18:
                pass
            else:
                plt.pause((1/18) - elapsed_time)

    def update(self):
        self.im.set_data(self.vol[:, self.ind, :, :].permute(1,2,0))
        self.im.axes.figure.canvas.draw()

def build_loop(dataset_name, NUM_VIDEOS, trainmode, logger):
    train_dataset = DBfactory(dataset_name, train=not trainmode, fold=1, config=cfg, is_human_trial=True)
    dl = DataLoader(train_dataset, 1, num_workers=0, shuffle=True, pin_memory=False)
    print(f'Dataloader has {len(dl)} samples!!!')

    all_labels = list(train_dataset.idx_to_class.values())
    count = 1
    for sample, lbl, path in dl:
        display_txt = f'Progress: {count}/{NUM_VIDEOS}'
        lbl_current = train_dataset.idx_to_class[lbl.item()]
        print(path[0])
        starttime = datetime.now()
        a = AppearanceFreeTrial(sample.squeeze(0), lbl_current, all_labels, display_txt, trainmode=trainmode)
        result_dict = a.res
        endtime = datetime.now()

        result_dict['Modality'] = dataset_name
        result_dict['Duration'] = (endtime - starttime).total_seconds()
        result_dict['StartTime'] = starttime
        result_dict['EndTime'] = endtime
        result_dict['UserID'] = logger.userID
        result_dict['Trainmode'] = int(trainmode)
        result_dict['VideoPath'] = path[0]

        logger.append(result_dict)
        logger.save()


        if count == NUM_VIDEOS:
            break
        count = count + 1


text0 = f'We investigate the ability to recognize actions from videos that have no texture.\n\
You will perform a multiple choice test, on such action videos.\n\
Your goal is to pick the action that you see being performed.\n\n'


text1 = f'This evaluation consists of 2 parts, taking about 5 and 15 minutes respectively.\n\
In the first part you will see regular videos, and will\n\
be asked to select what you see.\n\
In the second part you will be shown videos of the same actions, \n\
where the texture and shape has been removed.\n\
We are curious if you can solve the task.'

text2 = f'Before each evaluation there is a "warm-up", where you are shown the correct label.\n\n\
You may change the playback speed of the video with the slider\n\
below if it helps you in any way.\n\n\
Good Luck, and thank you for participating!'

def inbetween_modality_text(modality, num_practice_vids):
    if modality == "":
        txt = "Regular"
    elif modality == "rgbflow":
        txt = "Optical Flow"
    elif modality == 'af':
        txt = "Appearance Free"
    else: 
        raise ValueError(f"Something wrong, {modality=} not supported")

    return f"""
        The next modality is "{txt} Video".\n
        As a reminder: Warm-up will be {num_practice_vids} videos.\n
        The correct label is shown above the video during warm-up.\n\n
        Take a short break if you need it!

    """

if __name__ == "__main__":
    torch.manual_seed(69)
    ds_name = 'ucf5'
    userID = 'ECCV SCREENCAPTURE'
    global WINDOWTITLE
    WINDOWTITLE = 'Userstudy: ' + userID

    currentTime = str(datetime.now().replace(microsecond=0))
    savedir = f'trial/output/{currentTime}'
    logger = Logger(userID, savedir=savedir)

    NUM_VIDEOS = 10
    NUM_VIDEOS_PRACTICE = 1
    MessageScreen(text0)
    MessageScreen(text1)
    DemoActionsScreen("Continue", "ucf5")
    DemoActionsScreen("Continue", "afd5")
    CompositeScreen()


    # ------------------------ NORMAL VIDEO ---------------------
    MessageScreen(text2, 'Warm Up')

    modality = ""
    dataset_name = ds_name + modality
    # MessageScreen(inbetween_modality_text(modality, NUM_VIDEOS_PRACTICE))
    trainmode=True
    build_loop(dataset_name, NUM_VIDEOS_PRACTICE, trainmode, logger)
    MessageScreen("Warm-up is over. Start the evaulation!", "Start")
    trainmode=False
    build_loop(dataset_name, NUM_VIDEOS, trainmode, logger)

    MessageScreen("The first part is done! \n\n A quick refresher on Appearance Free Videos follows!", 'Continue')
    DemoActionsScreen("Continue", "afd5")
    # ------------------------ APPEARANCE FREE ---------------------
    # MessageScreen(text2, 'Warm Up')

    modality = "af"
    dataset_name = ds_name + modality
    # MessageScreen(inbetween_modality_text(modality, NUM_VIDEOS_PRACTICE))
    trainmode=True
    build_loop(dataset_name, NUM_VIDEOS_PRACTICE, trainmode, logger)
    MessageScreen("Warm-up is over. Start the evaulation!", "Start")
    trainmode=False
    build_loop(dataset_name, NUM_VIDEOS, trainmode, logger)

    MessageScreen("Thank you for participating. Grab some sweets you like :)", "Show Results")

    # ------------------------ SAVE STUFF ---------------------
    logger.save()
    visualize_plots(join(savedir, userID+'.csv'))
    print("Thank you for participating")