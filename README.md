# bacterial_genome

___________________________________________________________________
### General description
This repository can be used to predict the nucleosome, cohesine and polymerase coverage on bacterial genome intoduced in saccharomyces cerevisiae.

______________________________________________________________________
### Models availability
The models trained on sacchomyceres cerevisiae to predict the nucleosome, polymerase and cohesine coverage are available in Trainedmodels.

___________________________________________________________________________
### Data availability
The bacterial genomes are available in genome, both as fasta files and in one-hot encoded format in numpy binary files.
The labels used to train the models are available in data, in numpy binary files.

________________________________________________________________________

### Code reproducibility
The two main scripts are TRain_profile.py and predict_profile.py. Train_profile.py can be used to train a model from genome and labels. predict_profile.py can be used to perform predictions on a genome given a trained model. Genome and labels are expected as numpy binary files, genomes being already one-hot encoded. The helper scripts one_hot_encode.py, bw_to_npz.py and npz_to_bw.py can be used to convert between the different file formats.

Two environment files are provided: tf2.5cpu and tf2.5gpu. The second is simply an extension of the first allowing gpu acceleration on, through cudatoolkit and cudnn. This can be hardware dependent, so the tf2.5cpu environment is also provided. You will still be able to run the code, but it will be much slower on cpu. You may also install your own libraries on top of it to allow gpu acceleration.
The file bash_commands.sh shows commands used to train models and make predictions. The predictions aren't provided but running the prediction commands should provide the same results.

The two scripts Yeast_MNase_pipeline.py and Yeast_ChIP_pipeline.py were used to transform the MNase and ChIP files into a the labels files provided in data. These are provided as information, not necessarily to be rerun. If you wish to do so, you will need to download data from GSE217022. If you wish to apply them on your own data, you will probably need to adapt parts of the code (for example the remove_artifacts function).

Finally the notebook Figures.ipynb contains the code used to make figures for the article.
