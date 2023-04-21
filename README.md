# General concept

## **Abstract**
An application designed to help diagnose various medical conditions.

## **Repository structure**
- config  # (configurations of training and model parameters per model)
    - classification
        - ...
    - segmentation
        - ...
- datasets  # samples of datasets
- docs
- examples  # notebooks with examples, visualizations
- states  # (model states)
    - classification
    - segmentation
- src
    - diagnosisai
        - models  # (models aggregated by functionalities)
            - classification
                - ...
            - segmentation
                - ...
            - ...
        - utils
            - `data_processing.py`
            - `visualization.py`  # (plotly visualizations)
        - `functionals.py`  # (scoring, calculations etc.)
        - `custom_loss.py`  # (custom losses inherited from nn.Module)
        - `nn_datasets.py`  # (customs datasets inherited from torch.Datasets)
- tests  # (unit tests)


## Branch policies
There will be two main branches - `main` and `dev`. If you add new functionality or changes to the application, you should create a new branch with a name that informs you about the functionality or change. Once implemented, create a pull-request to dev and remove the sub-branch.

## Code standards
- Code should be written in PEP8 standard - [Python docs](https://peps.python.org/pep-0008/).
- Docstrings should be written using the reST standard - [Sphinx docs](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#info-field-lists) and [Python docs](https://peps.python.org/pep-0287/),


# Local environement and install
To install the package locally, go to the repository folder (where `setup.py` is) and type the following command:
```bash
pip install -e .
```
then it will be possible to import the package:
```python
import diagnosisai
```


