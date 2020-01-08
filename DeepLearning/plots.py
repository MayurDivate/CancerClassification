import os

import matplotlib.pyplot as plt


# plot the results

class Plotter:

    def __init__(self, outimg, outdir="./"):
        self.outimg = outimg
        self.outdir = outdir

    def plot_accuracy_and_loss(self, results):
        bname = str.replace(self.outimg, "_", " ")
        self.outimg = str.replace(self.outimg, " ", "_")
        outfile = os.path.join(self.outdir, self.outimg)

        # print(self.outimg)

        # loss plot
        plt.figure()
        plt.plot(results.history['loss'])
        plt.plot(results.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train', 'Validation'])
        plt.title('Loss: ' + bname)
        plt.savefig(outfile + "_Loss" + ".png")
        plt.close()

        # accuracy plot
        plt.figure()
        plt.plot(results.history['accuracy'])
        plt.plot(results.history['val_accuracy'])
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train', 'Validation'])
        plt.title('Accuracy: ' + bname)
        plt.savefig(outfile + "_Accuracy" + ".png")
        plt.close()
        print()
