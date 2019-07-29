import matplotlib.pyplot as plt
import pandas as pd


class KneePlot:

    def __init__(self, resfile, outfile):
        self.resfile = resfile
        self.outfile = outfile

    def knee_plot(self):
        df = pd.read_csv(self.resfile, sep='\t', header=None)
        df = df.iloc[:, [1, 2, 3]]
        df.columns = ['feat', 'acc', 'val_acc']
        df = df.sort_values('feat', ascending=True)
        df['feat'] = 1226 - df['feat']
        print(df.shape[0])

        plt.figure()
        plt.plot(df['feat'], df['acc'])
        plt.plot(df['feat'], df['val_acc'])
        #plt.xlim(1000,1300)
        plt.ylim(0.5, 1.2)
        plt.legend(['Accuracy', 'Val_accuracy'])
        plt.xlabel('features reduced by')
        plt.ylabel('')
        plt.savefig(self.outfile)

        plt.close()
