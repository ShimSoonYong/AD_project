import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from src.data import SELECT_NORMAL

class Scorer:
    def __init__(self, model, loader, threshold=0.01, criterion, device):
        self.model = model
        self.loader = loader
        self.THRESHOLDVAL = threshold # threshold val
        self.criterion = criterion
        self.DEVICE = device
        self.epoch = 0

    @torch.inference_mode()
    def visualize_IO(self, samples, num_samples=4):

        print("Making figures...")
        fig, ax =plt.subplots(2, num_samples, figsize=(num_samples, 2))

        for i in range(num_samples):
            img = samples.cpu().data.numpy()[i]
            a[0][i].imshow(img, cmap='gray')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())

            if(i == 0):
                a[0][i].set_ylabel('Input')

        for i in range(len(test_label)):
            img = decoded.cpu().data.numpy()[i]
            a[1][i].imshow(img, cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())

            if(i == 0): 
                a[1][i].set_ylabel('Output')

        plt.savefig("compare_io.png")


    @torch.inference_mode()
    def collect_record(self):
        record = {'id':[], 'label':[], 'score':[],'normal':[]}

        print("Collecting scores")
        count=0
        for step, (x, label) in enumerate(self.loader):

            print("Decoding samples to visualize")
            decoded = self.model(x)
            loss = float(self.criterion(decoded, x).cpu().numpy())

            self.visualize_IO(x)

            record['id'].append(step)
            record['label'].append(int(label!=SELECT_NORMAL))
            record['score'].append(loss) # anomaly score

            if loss > THRESHOLDVAL:
                record['normal'].append('0')
            else:
                record['normal'].append('1')

        self.record = record

    def plot_score(self):
        print("Visualizaing anomaly scores")
        label = np.array(self.record['label'])
        score = np.array(self.record['score'])

        score_min = score.min()
        score_max = score.max()

        plt.hist(score[np.where(label == 0)[0]], bins=30, range=(score_min, score_max), alpha=0.5, label='Normal')
        plt.hist(score[np.where(arr_label == 1)[0]], bins=30, range=(score_min, score_max), alpha=0.5, label='Abnormal')
        plt.xlabel("Anomaly score")
        plt.ylabel("Frequency")
        plt.axvline(self.THRESHOLDVAL, 0, 1, color='red', linestyle='--', linewidth=1)
        plt.legend(loc='upper right')
        plt.savefig("score_hist.png")

    def plot_auc(self):
        # Generating AUROC
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

        auroc = auc(fpr, tpr)
        print("AUROC: {}".format(auroc))

        print("Visualizing ROC curve")
        fpr, tpr, thresholds = roc_curve(self.record['label'], self.record['score'], pos_label=1)
        plt.plot(fpr, tpr)
        plt.title(f"AUROC: {auroc}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig("auroc.png")

    def save_record(self):
        self.record.pop('normal',None)
        self.record.pop('label',None)
        df = pd.DataFrame.from_dict(self.record)
        teamnumber = 2 # insert your teamnumber
        df.to_csv("result_team_{}.csv".format(teamnumber), index = False) # Index should be not included in the .csv file.

