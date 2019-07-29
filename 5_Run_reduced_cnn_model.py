from Interpret.Reduced.geneinfo import ReducedGeneInfo
from Interpret.Reduced.input import ReducedInput, HeatmapInput
from Interpret.Reduced.model import ReducedModel
from Interpret.Reduced.plots import KneePlot

raw_input_dir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/1_DL_data/CNN_CG'
input_tsv_dir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/2_Data_preprocessing/Input_tsv_files/reduced_cnn_cg'

resdir = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/3_Results/reduced_cnn'

original_geneinfo = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/DL_Cancer/Data/cg_ranked_list.tsv'
new_geneinfo = 'reduced_ranked_cnn_cg_list.tsv'

model_res = 'reduced_cnn_model_results.tsv'
labels = '/Users/mayurdivate/QUT/Work/Projects/1_CUP/1_DL_data/Labels.txt'


def run_reduced_model(total_features, reduce_by):
    mname = "CNN_CG_model_" + str(total_features - reduce_by)

    reduced_geneinfo = ReducedGeneInfo(original_geneinfo, reduce_by, new_geneinfo)
    reduced_geneinfo.create_reduce_geneinfo()

    reduced_input = ReducedInput(raw_input_dir, new_geneinfo, input_tsv_dir, labels, genetype='protein_coding')
    reduced_input.create_reduced_input()

    reduced_model = ReducedModel(input_tsv_dir, resdir, mname, (total_features - reduce_by))
    res = reduced_model.run_reduced_cnn_model()

    append_res([reduced_model.mname, reduced_model.nfeatures, res['accuracy'], res['val_accuracy'], res['loss'],
                res['val_loss']])



def append_res(res):
    with open(model_res, 'a') as f:
        for data in res:
            f.write(str(data) + "\t")
        f.write("\n")

        f.flush()


# run n model with different number of features each time

total_features = 19814

for reduce in range(19000, total_features, 100):
    print("reduced by: ", reduce)
    run_reduced_model(total_features, reduce)

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
