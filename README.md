# IUI Project - testing update of ML model 

## Dependencies

Easiest way of running is starting with conda environment [tensorflow-gpu](https://anaconda.org/anaconda/tensorflow-gpu).

`conda create -n tf_gpu tensorflow-gpu` 

As it (at the moment of writing) supports TF in version 2.6 and 2.7 is used, you need to manually update it. Moreover aforementioned environment does not include Jupyter, so you need to run following commands:

1. Activate created earlier environment: `conda activate tf_gpu`
2. Update tensorflow: `pip update -U tensorflow tensorflow-gpu`
3. Install missing dependencies: `pip install jupyter`

## Running

Create directory `./data` and place there some zipped training data. 

Scripts are sensitive to file names (you can change it in notebooks) and file structure in zips - it should look like this:

zip
- directory
  - class_name
    - images of class

To train models just run notebook, remembering, that `update_model_with_new_data` is using `initial_model` (from `initial_model` notebook) - so you have to run it first.

