# DVC initialization

- `pipenv install dvc`
- `dvc init`
- `dvc rmeote add --local <name> <url>`

Those params for authentication will be setup like that:

- `dvc remote modify --local <name> <param> <value>`

Be aware that we need to pass those values on the with the --local param. This is because doing this way we're not exposing sensible information on the repo.

- `dvc remote default --local <name>`

From now on you can add your data, or your models, or anything you want to track and run:

- `dvc add`: dvc will create a `<name_file_dir>.dvc`, and save a hash that correspond to that data.
- `dvc push`: will send your files to this remote repository (s3, google drive, gcp...)
- `dvc pull`: will bring all the data from a specific point in time, described on `.dvc` file

# Training

The general approach to train a DL model for object detection is:

- Initialization set up
- Load data and put on the correct format
- Finetune a pretrained model
- Predict
- Evaluation
- Save model and logs

Important to understand:

- how to load data into Dataset classes using pytorch

# Basic file structure:

The xml_to_csv script works if your data is on PASCAL VOC format. COCO format is given in a json file.

- data:
  - train
    - .jpg
    - .xml
  - test
    - .jpg
    - .xml
  - test_labels.csv
  - train_labels.csv
  - xml_to_csv.py
- outputs:
- src
  - config.py
  - datasets.py
  - inference.py
  - model.py
  - pipeline.py
  - utils.py
- test_data
- test_predictions
