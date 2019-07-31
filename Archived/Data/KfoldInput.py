from Datasets.InputData import InputData


class KfoldInput(InputData):

    def __init__(self, inputdirs, labels_tsv, geneinfo_tsv, genetype, outdir):
        self.inputdirs = inputdirs
        self.labels = self.get_labels(labels_tsv)
        self.geneinfo = geneinfo_tsv
        self.genetype = genetype
        self.outdir = outdir

    # start method for kfold class
    def create_k_inputs(self):
        print("Gene type:", self.genetype)

        for k in range(len(self.inputdirs)):
            kdir = self.inputdirs[k]
            ktsv = "K" + str(k + 1) + ".tsv"
            self.create_input_tsv(kdir, ktsv)
