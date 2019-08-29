from Datasets.InputData import InputData


class TrainTestValInput(InputData):

    def __init__(self, train_dir, test_dir, val_dir, labels_tsv, geneinfo, genetype, outdir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.labels = self.get_labels(labels_tsv)
        self.geneinfo = geneinfo
        self.genetype = genetype
        self.outdir = outdir

    def create_train_val_test_tsv(self):
        print("Gene type:", self.genetype)
        print(">>> Creating Training Dataset")
        self.create_input_tsv(self.train_dir, "Train.tsv")

        print(">>> Creating Validation Dataset")
        self.create_input_tsv(self.val_dir, "Val.tsv")

        print(">>> Creating Test Dataset")
        self.create_input_tsv(self.test_dir, "Test.tsv")







