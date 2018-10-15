# Image classifier

Image classifier is a supervised machine learning image classifier based on sklearn library and which uses Random Forest | k-nearest neighbor | Logistic Regression algorithms.

## Setting environment using python

    virtualenv env
    source env/bin/activate
    pip install -r requirements.txt

## Running the project using python

    python main.py --data-dir data_path --model randomForest

## Setting environment using nix ( you should have [nix](https://nixos.org/nix/download.html) installed )

    nix-shell -I nixpkgs=channel:nixos-18.03

## Running using nix

    nix-env -if default.nix
    image_classifier --data-dir data_path --model

`--data-dir` is the directory containing the data inputs/images. Images should have the format of `classname.*` ( for example `car.1.png` ).

`--model` is the machine learning algorithm to use. Three models are available `{ randomForest| knn | logisticRegression }`

## Making prediction

In python

    python main.py --predict image.1.jpg

In nix

    image_classifier --predict image.1.jpg