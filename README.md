# ``LADy``<sup>*</sup>: A System for Latent Aspect Detection
<sup>*Suggested by [Christine!](https://github.com/Lillliant)

1. [Colab Notebook](#1-Colab-Notebook)
2. [Structure](#2-Structure)
3. [Setup](#3-Setup)
4. [Quickstart](#4-Quickstart)
5. [Experiment](#5-Experiment)
6. [License](#6-License)

## 1. Colab Notebook
...
## 2. Structure

### Framework Structure

Sample outputs on [``semeval``](./data/raw/semeval/2016.txt) data can be seen here [``./output/semeval``](./output/semeval):
#### [``Year{K}``](./output/semeval/2016)
```
├── #Topics                                 -> Directory for aspect models with this number of aspects
|   ├── BTM                                 -> Directory for BTM model
|   ├── LDA                                 -> Directory for LDA model
|   ├── RND                                 -> Directory for Random model
├── reviews.pkl                             -> Reviews
├── splits.json                             -> Reviews converted to splits
```
#### [``btm``](./output/semeval/2016/25/btm)
```
├── f{k}.model.dict.pkl                      -> Dictionary of tokens/words for each fold K
├── f{k}.model.perf.cas                      -> Model's coherence for each fold K
├── f{k}.model.perf.perplexity               -> Model's perplexity for each fold K
├── f{k}.model.pkl                           -> BTM aspect model for each fold K
├── f{k}.model.train.log                     -> Log of the model during training
├── pred.eval.mean.csv                       -> Mean value of evaluations means in different folds
```
#### [``lda``](./output/semeval/2016/25/lda)
```
├── f{k}.model                               -> LDA aspect model for each fold K
├── f{k}.id2word                             -> Dictionary of tokens/words for each fold K
├── f{k}.model.perf.cas                      -> Model's coherence for each fold K
├── f{k}.model.perf.perplexity               -> Model's perplexity for each fold K                                   -> 
├── f{k}.model.train.log                     -> Log of the model during training
├── f{k}.pred.eval.mean.csv                  -> Mean value of evaluation means for each fold K
├── pred.eval.mean.csv                       -> Mean value of evaluations' means in different folds
```
#### [``rnd``](./output/semeval/2016/25/rnd)
```
├── f{k}.model.dict.pkl                      -> Dictionary of tokens/words for each fold K
├── f{k}.model.perf.cas                      -> Model's coherence for each fold K
├── f{k}.model.perf.perplexity               -> Model's perplexity for each fold K
├── f{k}.pred.eval.mean.csv                  -> Mean value of evaluation means for each fold K
├── pred.eval.mean.csv                       -> Mean value of evaluations means in different folds
```

### Code Structure
```
+---data
|   \---raw
|       \---semeval
|           |   2016.txt
|           |   toy.txt
|                       
+---output                 
+---src
|    |   main.py
|    |   params.py
|    |   __init__.py
|    |   
|    +---aml
|    |   |   btm.py
|    |   |   lda.py
|    |   |   mdl.py
|    |   |   rnd.py
|    |           
|    +---cmn
|    |   |   review.py
|    |   |   semeval.py 
|    |
+--- .gitignore
+--- environment.yml
+--- README.md
+--- license.txt
\--- requirements.txt
```

## 3. Setup

It has been developed on `Python 3.8` and can be installed by `conda` or `pip`:

```bash
git clone https://github.com/fani-lab/LADy.git
cd LADy
conda env create -f environment.yml
conda activate lady
```

```bash
git clone https://github.com/fani-lab/LADy.git
cd LADy
pip install -r requirements.txt
```

This command installs compatible versions of the following libraries:

>* aml: ``gensim, nltk, pandas, requests, bitermplus``
>* others: ``pyLDAvis, pytrec-eval-terrier, sklearn, numpy, scipy, matplotlib``


Additionally, you need to install the following libraries from their source:
- [``Microsoft C++ Build Tools``](https://visualstudio.microsoft.com/visual-cpp-build-tools/) as a requirement in ``btm``.
- ``en_core_web_sm`` as a requirement in ``spaCy`` library with the following command:
  
  ```bash
  python -m spacy download en_core_web_sm
  ```
  
- ``stopwords`` as a requirement in ``nltk`` library with the following command:
  
  ```bash
  python -m nltk.downloader stopwords
  ```

## 4. Quickstart

### Data

We have semeval datasets for 2015 and 2016 at [`./data/raw/semeval`](./data/raw/semeval).

For quickstart purposes, a `toy` sample of reviews has been provided at [`./data/raw/semeval/toy.txt`](./data/raw/semeval/toy.txt).


### Run
You can run the framework via [`./src/main.py`](./src/main.py) with following command:
```bash
python src/main.py --aml lda rnd btm --data "data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml" --output "output/semeval/2016" --naspects 25
```
where the input arguements are:

`aml`: A list of aspect modeling methods among {`lda`, `rnd`, `btm`}, required, case-insensitive.
`data`: Raw dataset file path, required.
`output`: Output path, required.
`naspects`: number of aspects, required.

A run will produce an output folder at `./output/semeval` and sub folders for each aspect modeling pair as baselines, e.g., `lda`, `rnd`, and `btm`. The final evaluation results are aggregated in `./output/semeval/2016/25/btm/pred.eval.mean.csv`,`./output/semeval/2016/25/lda/pred.eval.mean.csv`, and `./output/semeval/2016/25/rnd/pred.eval.mean.csv` . See an example run on semeval dataset at [`./output/semeval`](./output/semeval). 


## 5. Experiment
Due to OOV (an aspect might be in test set which is not seen in traning set during model training), we may have metric@n for n >> +inf not equal to 1.

## 6. License
©2021. This work is licensed under a [CC BY-NC-SA 4.0](LICENSE.txt) license.

### Authors
Farinam Hemmati Zadeh <sup>1,2</sup>, [Hossein Fani](https://hosseinfani.github.io/)<sup>1,3</sup>

<sup><sup>1</sup>School of Computer Science, Faculty of Science, University of Windsor, ON, Canada.</sup>

<sup><sup>2</sup>[hemmatif@uwindsor.ca](mailto:hemmatif@uwindsor.ca), [f.hemmatizadeh@gmail.com](mailto:f.hemmatizadeh@gmail.com)</sup>
<sup><sup>3</sup>[hfani@uwindsor.ca](mailto:hfani@uwindsor.ca)</sup>

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### Acknowledgments
In this work, we use [``bitermplus``](https://github.com/maximtrp/bitermplus), [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval), and other libraries. We would like to thank the authors of these libraries.
