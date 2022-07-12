import io

import torch
import PIL.Image
import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

class ConfusionMatrix(object):
    def __init__(self, n_classes, labels=None):
        self.n_classes = n_classes
        self.mat = torch.zeros(n_classes, n_classes, requires_grad=False)
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.mat[t, p] += 1

    def reset(self):
        self.mat = torch.zeros(self.n_classes, self.n_classes, requires_grad=False)

    def _create_figure(self, display_values, normalize, fontsize=5, label_angle=45):
        # create a nice looking confusion matrix with matplotlibs matshow
        # and then create an image from it, so it can be
        # for example shown in tensorboard

        mat = self.mat

        if normalize:
            total_samples = mat.sum(axis=1)
            mat = mat / total_samples[:, None]

        # Sanity check: Do the rows sum to 1.0?
        # print(mat.sum(axis=1))

        # plt.matshow(mat, cmap=plt.cm.Blues)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(mat, cmap=plt.cm.Blues)

        if display_values:
            threshold = mat.max() / 2.
            for (i, j), z in np.ndenumerate(mat):
                # this puts white bordera round the text, but i like the version where the text is either
                # black or white depending on the z value
                # ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center', path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")], fontsize=5)
                color = "white" if z > threshold else "black"
                ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center', color=color, fontsize=fontsize)

        if self.labels is not None:
            ax.set_xticks(range(0, self.n_classes))
            ax.set_xticklabels(self.labels, rotation=label_angle, fontsize=fontsize)
            ax.set_yticks(range(0, self.n_classes))
            ax.set_yticklabels(self.labels, fontsize=fontsize)

        ax.set_ylabel('True Class')
        ax.set_xlabel('Predicted Class')
        fig.colorbar(cax)

        return fig

    def _fig_to_img(self, fig, dpi):
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, format='png')
        buf.seek(0)
        image = PIL.Image.open(buf).convert('RGB')
        image = ToTensor()(image)
        plt.close(fig)
        return image

    def as_img(self, display_values=True, normalize=True, dpi=800, fontsize=5, label_angle=45):
        fig = self._create_figure(display_values, normalize, fontsize, label_angle)
        image = self._fig_to_img(fig, dpi)
        return image


if __name__ == '__main__':
    cm = ConfusionMatrix(3, list(range(3)))
    print(f'{cm.n_classes}, {cm.labels}')
    allfig, ax = plt.subplots(2,3)
    ax = ax.flatten()

    for i in range(3):
        print(f'in iter {i}')
        predictions_random = np.floor(3 * np.random.rand(10)).astype(int)
        # predictions_random = [i] * 10
        labels_uniform = np.floor(3 * np.random.rand(10)).astype(int)
        cm.update(predictions_random, labels_uniform)
        I = cm.as_img(dpi=250, display_values=True, normalize=False).permute(1,2,0)
        ax[i].imshow(I)
        ax[i].axis('off')

        H = cm.as_histogram_img(dpi=250).permute(1,2,0)
        ax[i+3].imshow(H)
        ax[i+3].axis('off')

    plt.show()
