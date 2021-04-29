<p align="center">
	<a>
		<img src="https://yotta-academy.com/wp-content/uploads/2019/10/logo.svg"
             width="50%" alt="yotta">
	</a>
</p>


# Galaxy Zoo
<p align="center">
	<a href="https://www.python.org/downloads/release/python-380/">
		<img src="https://img.shields.io/badge/python-3.8-blue"
			 alt="Python Version">
	</a>
	<a href="https://pytorch.org/">
		<img src="https://img.shields.io/badge/framework-pytorch-red"
			 alt="License">
	</a>
	<a href="https://gitlab.com/yotta-academy/mle-bootcamp/projects/dl-projects/project-2-winter-2021/galaxy.zoo_simon_dan_jeremie/-/blob/master/LICENSE.txt">
		<img src="https://img.shields.io/badge/license-BSD-green"
			 alt="License">
	</a>
	<a href="https://github.com/psf/black">
		<img src="https://img.shields.io/badge/code%20style-black-000000.svg"
			 alt="Code Style">
	</a>
	<a href="https://gitmoji.dev">
		<img src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg"
			 alt="Gitmoji">
	</a>
</p>

This project aims to classify the morphologies of distant galaxies using deep neural networks.

It is based on the Kaggle [Galaxy Zoo Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview).

<p align="center">
	<a>
		<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/M104_ngc4594_sombrero_galaxy_hi-res.jpg/1200px-M104_ngc4594_sombrero_galaxy_hi-res.jpg"
             width="100%">
	</a>
</p>

## Documentation
Project's assignement as well as inspirational papers on the topic are available in [doc/](doc/).

To better understand the task to be learned, you could give it a go yourself ! [try it here](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/classify)


## Installation
1. (Optional) Install `poetry` if you don't have it already:
```bash
make setup-poetry
```

2. Install dependencies:
```bash
poetry install
```

3. To download the dataset, you can install [Kaggle's API](https://github.com/Kaggle/kaggle-api) (you need to setup your credentials), and then download the dataset:
```bash
pip install --user kaggle
kaggle competitions download -c galaxy-zoo-the-galaxy-challenge
```

4. You're good to go!


## Train
### Create the training labels for classification
```bash
poetry run python -m gzoo.app.make_labels <data_dir>
```
required arguments:
- `<data_dir>`: specifies the location of the dataset directory containing the original regression labels `training_solutions_rev1.csv`

### Run the classification pipeline:
```bash
poetry run python -m gzoo.app.train -o config/train_classification.yaml
```
script option:
- `-o`: specify the `.yaml`config file to read options from.
Every run config option should be listed in this file (the default file for this is [config/train_classification.yaml](config/train_classification.yaml)) and every option in the `yaml` file can be overloaded *on the fly* at the command line.

For instance, if you are fine with the values in the `yaml` config file but you just want to change the `epochs` number, you can either change it in the config file *or* you can directly run:
```bash
poetry run python -m gzoo.app.evaluate -o config/train.yaml --epochs 50
```
This will use all config values from `config/train.yaml` except the number of epochs which will be set to `50`.

main run options:
- `--seed`: seed for initializing training. (default: `None`)
- `--epochs`: total number of epochs (default: `90`)
- `--batch-size`: batch size (default: `256`)
- `--workers`: number of threads (default: `4`)
- `--model.arch`: model architecture to be used(default: `resnet18`)
- `--model.pretrained`: use pre-trained model (default: `False`)
- `--optimizer.lr`: optimizer learning rate (default: `3.e-4` with Adam)
- `--optimizer.momentum`: optimizer momentum (default: `0.9`)
- `--optimizer.weight-decay`: optimizer weights regularization (L2) (default `1.e-4`)


## Predict
### From the web app
```bash
streamlit run gzoo/interface/web_app.py
```

### From the command line:
```bash
poetry run python -m gzoo.app.predict -o config/predict.yaml
```
Config works the same as for `train.py`, default config is at [config/predict.yaml](config/predict.yaml).
The `dataset` directory specified in the config must contain an `images_test_rev1` that contains itself the images to predict, as well as the `all_ones_benchmark.csv` output template from the Kaggle project's data sources.

A 1-image example is provided which you can run with:
```bash
poetry run python -m gzoo.app.predict -o config/predict.yaml --dataset example
```

## Developer
Activate pre-commit hooks:
```bash
poetry run pre-commit install
```
