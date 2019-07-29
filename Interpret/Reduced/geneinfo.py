import pandas as pd


class ReducedGeneInfo:

    def __init__(self, original_geneinfo, reduce_by, new_geneinfo):
        self.original_geneinfo = original_geneinfo
        self.reduce_by = reduce_by
        self.new_geneinfo = new_geneinfo

    def create_reduce_geneinfo(self):
        print("Reduce by = ", self.reduce_by)
        df = pd.read_csv(self.original_geneinfo, sep='\t')
        df = df[:(df.shape[0] - self.reduce_by)]
        # print(df.shape)
        df.to_csv(self.new_geneinfo, sep='\t', index=False)
