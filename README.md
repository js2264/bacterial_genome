# bacterial_genome

___________________________________________________________________
### General description
This repository can be used to predict the nucleosome, cohesine and polymerase coverage on bacterial genome intoduced in saccharomyces cerevisiae.

______________________________________________________________________
### Models availability
The models trained on sacchomyceres cerevisiae to predict the nucleosome, polymerase and cohesine coverage are available in models.


___________________________________________________________________________
### Data availability
The bacterial genome in fasta file are available in data. 

________________________________________________________________________

### Code reproducibility
A notebook per protein is available, use a notebook to predict the protein coverage on sacchomyceres cerevisiae geneome, on bacterial genome and to generate analysis figure.

Use the script training.py to train a model to predict cohesine or pol2 on saccharomyceres cerevisiae. You need to specify the configuration for the training session in configuration.yml (`annotation_file` stands for the file containing the protein coverage, `annotation_type`can be either 'pol2' or 'cohesine' and `file_to_store_model` stands for the output file)
