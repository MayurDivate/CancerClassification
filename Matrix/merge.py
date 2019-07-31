import pandas as pd


class GeneBySampleMerger:
    """
    It merges two or more gene by sample matrices together with fillna = 0 .
    Index name must me 'gene_id'.
    """

    def __init__(self, m_files, outfile):
        """
        :param m_files:
        :param outfile:
        """
        if len(m_files) < 2:
            exit()

        self.m_files = m_files
        self.outfile = outfile

    def merge(self):

        m = pd.read_csv(self.m_files[0], sep='\t', index_col='gene_id')
        print(self.m_files[0])

        for m_file in self.m_files[1:]:
            m2 = pd.read_csv(m_file, sep='\t', index_col='gene_id')
            m = m.merge(m2, how='outer', on='gene_id').fillna(0)
            print(m_file)

        m.to_csv(self.outfile, sep='\t')
