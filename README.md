# Numerai Easy ML

_A spin off of [cookie-cutter data science](https://github.com/drivendata/cookiecutter-data-science) from [reproducible-ml](https://github.com/carlomazzaferro/reproducible-ml) (under construction) applied to the numer.ai data science competition._

Project Organization
--------------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Other data, if needed
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump from the numerai website
    │
    ├── models             <- Trained and serialized models, dumped from TensorFlow, sklearn, etc.
    │
    ├── notebooks          <- Jupyter notebooks for exploration
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── model_ouputs   <- Output of models, which include: ROC curve img, FPR, TPR, paraneters in JSON file,
    │        │                submission file, validation predictions.
    │        ├── MODEL_0
    │        ├── MODEL_1
    │         ...
    │        └── MODEL_N
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling (empty)
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions. Each model spcifies either a Neural Net architecture,
        │   │                 sklearn algorithm, etc.
        │   ├── base.py    <- A general purpose Base and Project class from which the MODEL_N inherits
        │   │                 from, where a variety of useful methods are implemented.
        │   ├── MODEL_1.py
        │   ├── MODEL_2.py
        │   ├── MODEL_3.py
        │   ├── MODEL_4.py
        │   └── __init__.py 
        │     
        └── utils
            ├── __init__.py 
            ├── definitions.py  <- File structure definitions
            └── utilities.py    <- General purpose scripts
    



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
