# Image classifier

Image classifier is a supervised machine learning image classifier based on sklearn and uses Random Forest | k-nearest neighbor | Logistic Regression.

## Running the project using python

    virtualenv env
    source env/bin/activate
    pip install -r requirements.txt
    python main.py --data-dir data_path --model randomForest

## Running using nix ( you should have [nix](https://nixos.org/nix/download.html) installed )

    nix-env -if default.nix
    image_classifier --data-dir data_path --model

`--data-dir` is the directory containing the data inputs/images. Images should have the format of `classname.*` ( for example `car.1.png` ).

`--model` is the machine learning algorithm to use. Three models are available `{ randomForest| knn | logisticRegression }`