# P1_P2_detection_ratio

Since it is not possible to publicly share intracranial pressure signals from real patients, a synthetic toy dataset is provided as an example to run the analysis pipeline. Therefore, the results obtained with it have no scientific significance. 
Training pipelines used to select NN architectures both for pulse selection and P1/P2 detection are provided in two separate folders.

To run a pipeline, cd to the corresponding folder(P1_P2_detection or calculable_ratio) then:

$ conda env create -f workflow/envs/venv.yaml
$ conda activate venv
$ cd workflow
$ snakemake -s Snakefile --cores 8
