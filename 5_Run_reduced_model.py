from Interpret.Reduced.geneinfo import ReducedGeneInfo
from Interpret.Reduced.input import ReducedInput, HeatmapInput
from Interpret.Reduced.model import ReducedModel
from Interpret.Reduced.plots import KneePlot

raw_input_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/1_DL_data/lincRNA'
input_tsv_dir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/3_Data_preprocessing/ReducedInputTsvFiles'

resdir = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/4_Results/lincRNA_MLP_reduced_model'

original_geneinfo = '/Users/n10337547/Projects/1_CUP/1_Blood_Cancer/DL_Cancer/Data/lincRNA_ranked_list.tsv'
new_geneinfo = 'reduced_ranked_lincRNA_list.tsv'

model_res = 'reduced_model_results.tsv'


def run_reduced_model(total_features, reduce_by):
    mname = "MLP_model_" + str(total_features - reduce_by)

    reduced_geneinfo = ReducedGeneInfo(original_geneinfo, reduce_by, new_geneinfo)
    reduced_geneinfo.create_reduce_geneinfo()

    reduced_input = ReducedInput(raw_input_dir, new_geneinfo, input_tsv_dir)
    reduced_input.create_reduced_input()

    reduced_model = ReducedModel(input_tsv_dir, resdir, mname, (total_features - reduce_by))
    res = reduced_model.run_reduced_mlp_model()

    append_res([reduced_model.mname, reduced_model.nfeatures, res['accuracy'], res['val_accuracy'], res['loss'],
                res['val_loss']])


def append_res(res):
    with open(model_res, 'a') as f:
        for data in res:
            f.write(str(data) + "\t")
        f.write("\n")

        f.flush()


# run n model with different number of features each time

total_features = 1226

for reduce in range(1000, 1100, 1):
    # run_reduced_model(total_features, reduce)
    break


# plot results
def kplot():
    kneeplot = KneePlot(model_res, 'model_res.png')
    kneeplot.knee_plot()


# kplot()



# heatmap input
def get_heatmap_input(total_features, reduce_by):
    print('creating heatmap input')
    reduced_geneinfo = ReducedGeneInfo(original_geneinfo, reduce_by, new_geneinfo)
    reduced_geneinfo.create_reduce_geneinfo()

    reduced_input = ReducedInput(raw_input_dir, new_geneinfo, input_tsv_dir)
    reduced_input.create_reduced_input()
    heatmap_input = HeatmapInput(input_tsv_dir)
    heatmap_input.create_heatmp_input_tsv()

# get_heatmap_input(total_features, 1070)
