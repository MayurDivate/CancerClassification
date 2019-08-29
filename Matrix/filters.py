import pandas as pd


class GeneFilter:
    """
    Takes gene x sample matrix, creates the subset of matrix by discarding gene which
    are not in the gene_list_file.

    """

    def __init__(self, matrix_file, gene_list_file, outfile):
        self.matrix_file = matrix_file
        self.gene_list_file = gene_list_file
        self.outfile = outfile

    def filter(self):
        matrix = pd.read_csv(self.matrix_file, sep='\t', index_col='gene_id')
        genes = pd.read_csv(self.gene_list_file, sep='\t')
        matrix = matrix.reindex(genes['gene_id']).dropna(how='all')
        print(matrix.info())
        matrix.to_csv(self.outfile, sep='\t')
        return matrix


class SampleFilter:
    """
    create the subset of the matrix by sample names
    """

    def __init__(self, matrix_file, sample_list_file, outfile):
        self.matrix_file = matrix_file
        self.sample_list_file = sample_list_file
        self.outfile = outfile

    def filter(self):
        matrix = pd.read_csv(self.matrix_file, sep='\t', index_col='gene_id')
        samples = [line.rstrip() for line in open(self.sample_list_file, 'r')]
        matrix = matrix[samples]
        matrix.to_csv(self.outfile, sep='\t')
        print(matrix.info())
        return matrix


class MatrixFilter:

    def __init__(self, matrix_file, name_list_file, outfile):
        self.matrix_file = matrix_file
        self.name_list_file = name_list_file
        self.outfile = outfile

    def get_name_list(self):
        return [line.rstrip() for line in open(self.name_list_file, 'r')]


class RowFilter(MatrixFilter):

    def apply_filter(self):
        matrix = pd.read_csv(self.matrix_file, sep='\t', index_col=0)

        if isinstance(self.name_list_file, list):
            row_ids = self.name_list_file
        else:
            row_ids = self.get_name_list()

        matrix = matrix.reindex(row_ids).dropna(how='all')
        print(matrix.info())
        matrix.to_csv(self.outfile, sep='\t')
        return matrix


class ColumnFilter(MatrixFilter):

    def apply_filter(self):
        matrix = pd.read_csv(self.matrix_file, sep='\t', index_col=0)
        if isinstance(self.name_list_file, list):
            column_ids = self.name_list_file
        else:
            column_ids = self.get_name_list()

        matrix = matrix[column_ids]
        print(matrix.info())
        matrix.to_csv(self.outfile, sep='\t')
        return matrix
