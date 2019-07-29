import os

import pandas as pd


class InputData:

    def __init__(self, inputdir, labels_tsv, geneinfo, genetype, outdir):
        self.inputdir = inputdir
        self.labels = self.get_labels(labels_tsv)
        self.geneinfo = geneinfo
        self.genetype = genetype
        self.outdir = outdir

    # returns df of labels for each expression file
    def get_labels(self, input_file):
        with open(input_file, 'r') as lf:
            df = pd.read_csv(lf, sep="\t", header=-1)
            df.columns = ['file', 'label']
            df.index = list(df['file'])
            return df

    # returns list of files in in the indicated directory
    def get_FPKM_files(dir):
        return [f for f in os.listdir(dir) if not f.startswith('.')]

    # get the label for the input file
    def get_file_label(self, f):
        f = f.replace(".FPKM.txt", "")
        return self.labels.loc[f, 'label']

    # get the FPKM values from given file
    def get_FPKM_Values(self, f):
        with open(f, 'r') as fileX:
            df = pd.read_csv(fileX, sep='\t', header=None)
            filename = self.get_basename(f).replace(".FPKM.txt","")
            df.columns =  ['Genes', filename]
            return self.get_gene_type_data(df)

    def get_basename(self,f):
        return os.path.basename(f)

    # filter the FPKM values according to the genetype
    def get_gene_type_data(self, df):

        with open(self.geneinfo, 'r') as geneinfo:
            dfX = pd.read_csv(geneinfo, sep='\t')
            dfX = dfX.iloc[:, [0, 2]]
            df = pd.merge(df, dfX, left_on='Genes', right_on='gene_id')
            df = df[df['gene_type'] == self.genetype]
            df = df.sort_values(by=['Genes'])
            df = df.reset_index(drop=True)
            return df.iloc[:, 0:2]

    # put all expression values in file
    def create_input_tsv(self, dirname, outfile):
        print("directory:", dirname)
        fpkm_files = InputData.get_FPKM_files(dirname)
        outdf = pd.DataFrame()

        for f in fpkm_files:
            flabel = self.get_file_label(f)
            df = self.get_FPKM_Values(os.path.join(dirname, f))
            header = list(df.columns)
            df = df.append(pd.DataFrame({header[0]: ['label'], header[1]: [flabel]}))
            if outdf.size == 0:
                outdf = df
            else:
                outdf = outdf.merge(df, on='Genes')

        outdf = outdf.T
        print(outdf.info())
        outdf.to_csv(os.path.join(self.outdir, outfile), sep='\t', index=False)
        print("--done--")
