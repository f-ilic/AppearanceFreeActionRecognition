from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torchvision.io import write_video, read_video
import torchvision
torchvision.set_video_backend("video_reader")

class VideoTensorViewer(object):
    def __init__(self, vol, wrap_around=True, figsize=(8,6)):
        # vol has to be shape C,T, H, W
        self.vol = vol
        self.wrap_around = wrap_around
        self.slices = vol.shape[1]
        self.ind = 0
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.im = self.ax.imshow(self.vol[:, self.ind, ...].permute(1,2,0))
        self.update()
        plt.title(f'SCROLL UP/DOWN')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def onscroll(self, event):
        if event.button == 'up':
            if not self.wrap_around and self.ind+1 == self.slices:
                return
            self.ind = (self.ind + 1) % self.slices
        else:
            if not self.wrap_around and self.ind == 0:
                return
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.vol[:, self.ind, :, :].permute(1,2,0))
        self.im.axes.figure.canvas.draw()


if __name__ == "__main__":
    # path = '/home/f/datasets/ucf101rgbflow/BlowDryHair/v_BlowDryHair_g10_c04.avi'
    # path = '/home/f/datasets/ucf101rgbflow/Knitting/v_Knitting_g11_c04.avi'
    # path = '/home/f/datasets/ucf101rgbflow/PlayingViolin/v_PlayingViolin_g10_c03.avi'
    path = '/home/f/datasets/ucf101rgbflow/Fencing/v_Fencing_g09_c06.avi'
    x = read_video(path)[0].permute(3,0,1,2)
    VideoTensorViewer(x)
    plt.show(block=True)
