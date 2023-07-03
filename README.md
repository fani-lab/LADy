# 💃 `LADy`<sup>*</sup>: A System for Latent Aspect Detection
<sup>*Suggested by [Christine!](https://github.com/Lillliant)


`LADy` is a `python-based` framework to facilitate research in `aspect detection`, which involves extracting `aspects` of products or services in reviews toward which customers target their opinions and sentiments. Aspects could be explicitly mentioned in reviews or be `latent` due to social background knowledge. With a special focus on `latent aspect detection `, `LADy` hosts various canonical aspect detection methods and benchmark datasets of unsolicited reviews from `semeval` and `google reviews`.
`LADy`'s object-oriented design allows for easy integration of new `methods`, `datasets`, and `evaluation metrics`. Notably, `LADy` features review `augmentation` via `natural language backtranslation` that can be seamlessly integrated into the training phase of the models to boost `efficiency` and improve `efficacy` during inference.

<table align="center" border=0>
<tr>
<td >

- [1. Setup](#1-setup)
- [2. Quickstart](#2-quickstart)
- [3. Structure](#3-structure)
- [4. Experiment](#4-experiment)
- [5. License](#5-license)
- [6. Acknowledgments](#6-acknowledgments)
- [7. Awards](#7-awards)
- [8. Contributing](#8-contributing)

</td>
<td>
  <p align="center">
    <img src='./src/aml/LADy.png' width="550">
<!--     <br> -->
<!--     <a href="https://lucid.app/lucidchart/fe256064-3fda-465a-9abc-036dfc40acad/edit?view_items=svRVuxyZvY9n%2CsvRVVLD91NpJ%2CxDRV-pti53Ae%2CwJRVh7la6C-y%2CBLRV4aXmE.uY%2CBLRVOyM~DMFW&invitationId=inv_6e8aa9a6-1854-4ecf-a753-e1b2e05b50fc">class diagram for aspect modeling hierarchy</a> -->
  </p>
</td>
</tr>
</table>


## 1. Setup
`LADy` has been developed on `Python 3.8` and can be installed by `conda` or `pip`:

```bash
git clone --recursive https://github.com/fani-lab/LADy.git
cd LADy
conda env create -f environment.yml
conda activate lady
```

```bash
git clone --recursive https://github.com/fani-lab/LADy.git
cd LADy
pip install -r requirements.txt
```
This command installs compatible versions of the following libraries:

> [`./src/cmn`](./src/cmn): `transformers, sentence_transformers, scipy, simalign, nltk`

> [`./src/aml`](./src/aml): `gensim, nltk, pandas, requests, bitermplus, contextualized_topic_models`

> others: `pytrec-eval-terrier, sklearn, matplotlib, seaborn, tqdm`

Additionally, the following libraries should be installed:
> [`Microsoft C++ Build Tools`](https://visualstudio.microsoft.com/visual-cpp-build-tools/) as a requirement of biterm topic modeling in [`./src/btm.py`](./src/btm.py).

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

Further, we reused [`octis`](https://aclanthology.org/2021.eacl-demos.31.pdf) as submodule [`./src/octis`](https://github.com/fani-lab/OCTIS) for `unsupervised` neural aspect modeling using e.g., [`neural lda`](https://arxiv.org/pdf/1703.01488.pdf):

```bash
cd src/octis
python setup.py install
```

Finally, we added below state-of-the-art aspect detection baselines as submodules:
> [`bert-e2e-absa`](https://aclanthology.org/D19-5505/) → [`./src/bert-e2e-absa`](https://github.com/fani-lab/BERT-E2E-ABSA)

> [`hast`](https://www.ijcai.org/proceedings/2018/0583) → [`./src/hast`](https://github.com/fani-lab/HAST)

> [`cat`](https://aclanthology.org/2020.acl-main.290/) → [`./src/cat`](https://github.com/fani-lab/cat)

## 2. Quickstart[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fani-lab/LADy/blob/main/quickstart.ipynb)
For quickstart purposes, a `toy` sample of reviews has been provided at [`./data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml`](./data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml).
You can run `LADy` by:
```bash
cd ./src
python main.py -naspects 5 -am rnd -data ../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml -output ../output/toy.2016SB5/
```
This run will produce an output folder at [`../output/toy.2016SB5/`](./output/toy.2016SB5/) and a subfolder for `rnd` aspect modeling (random) baseline.
The final evaluation results are aggregated in [`../output/toy.2016SB5/agg.pred.eval.mean.csv`](./output/toy.2016SB5/agg.pred.eval.mean.csv). 

## 3. Structure
`LADy` has two layers: 

### [`./src/cmn`](./src/cmn)
Common layer (`cmn`) includes the abstract class definition for `Review`. 
Important attributes of `Review` are:

> `self.aos`: stores a list of `(aspect, opinion, sentiment)` triples for each sentence of a review, and 

> `self.augs`: stores the translated (`Review_`) and back-translated (`Review__`) versions of the original review along with the semantic similarity of back-translated version with original review in a dictionay `{'lang': (Review_, Review__, semantic score)}`

> `self.parent`: whether `self` is an original review or a translated or back-translated version.

This layer further includes `SemEvalReview`, which is a realization of `Review` class for reviews of `SemEval` datasets.
Specifically, this class overrides loading `SemEval`'s reviews into `Review` objects and stores it into a pickle file after preprocessing.
Pickle file is later used by models for training and testing purposes. Sample pickle files for a `toy` dataset: [`./output/semeval+/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml`](./output/semeval+/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml), there are some
where the filename `review.{list of languages}.pkl` shows the review objects also include back-translated versions in `{list of languages}`.

<p align="center">
 <img src='./src/cmn/LADy.png' width="550" >
 <br>
 <a href="https://lucid.app/lucidchart/fe256064-3fda-465a-9abc-036dfc40acad/edit?view_items=svRVuxyZvY9n%2CsvRVVLD91NpJ%2CxDRV-pti53Ae%2CwJRVh7la6C-y%2CBLRV4aXmE.uY%2CBLRVOyM~DMFW&invitationId=inv_6e8aa9a6-1854-4ecf-a753-e1b2e05b50fc">class diagram for review</a>
</p>

### [`./src/aml`](./src/aml)
Aspect model layer (`aml`) includes the abstract class definition `AbstractAspectModel` for aspect modeling methods. 
Important methods of are:

> `self.train(reviews_train, reviews_valid, ..., output)`: train the model on input training and validation samples and save the model in `output`,  

> `self.infer(review)`: infer (predict) the aspect of a given review, which is an ordered list of `self.naspect` aspects with different probability scores, like `[(0, 0.7), (1, 0.1), ...]`
To view the actual aspect terms (tokens), `self.get_aspect_words(aspect_id)` can be used which returns an ordered list of terms with probability scores like `[('food', 0.4),('sushi', 0.3), ...]`

> `self.load(path)`: loads a saved trained model.

This layer further includes realizations for different aspect modeling methods like, 

Finally, we added other recourses as our baselines which are [`BERT-E2E-ABSA`](https://aclanthology.org/D19-5505/) as a [`submodule`](https://github.com/fani-lab/BERT-E2E-ABSA), [`HAST`](https://www.ijcai.org/proceedings/2018/0583) as a [`submodule`](https://github.com/fani-lab/HAST), and [`CAt`](https://aclanthology.org/2020.acl-main.290/) as as a [`submodule`](https://github.com/fani-lab/cat) for aspect detection.

> [`Local LDA [Brody and Elhadad, NAACL2010]`](https://aclanthology.org/N10-1122/) in [`./src/aml/lda.py`](./src/aml/lda.py),

> [`Biterm Topic Modeling [WWW2013]`](https://dl.acm.org/doi/10.1145/2488388.2488514) in [`./src/aml/btm.py`](./src/aml/btm.py),

> [`Contextual Topic Modeling [EACL2021]`](https://aclanthology.org/2021.eacl-main.143/) in [`./src/aml/ctm.py`](./src/aml/ctm.py),

> [`BERT-E2E-ABSA [W-NUT@EMNLP2019]`](https://aclanthology.org/D19-5505/) to be added,
> 
> [`HAST [IJCAI2018]`](https://aclanthology.org/2021.eacl-main.143/) to be added,
> 
> [`CAt [ACL2020]`](https://aclanthology.org/2020.acl-main.290/) to be added,
> 
> [`Random`](./src/aml/rnd.py) in [`./src/aml/ctm.py`](./src/aml/rnd.py), which returns a shuffled list of tokens as a prediction for aspects of a review to provide a minimum baseline for comparison.

Sample models trained on a `toy` dataset can be found [`./output/toy.2016SB5//{model name}`](./output/toy.2016SB5/).

<p align="center"><img src='./src/aml/LADy.png' width="550" >
 <br>
  <a href="https://lucid.app/lucidchart/fe256064-3fda-465a-9abc-036dfc40acad/edit?view_items=svRVuxyZvY9n%2CsvRVVLD91NpJ%2CxDRV-pti53Ae%2CwJRVh7la6C-y%2CBLRV4aXmE.uY%2CBLRVOyM~DMFW&invitationId=inv_6e8aa9a6-1854-4ecf-a753-e1b2e05b50fc">class diagram for aspect modeling hierarchy</a>
</p>

### [`./src/main.py`](./src/main.py)
`LADy`'s driver code accepts the following args:

> `-naspects`: the number of possible aspects for a review in a domain, e.g., `-naspect 5`, like in `restaurant` we may have 5 aspects including `['food', 'staff', ...]`

> `-am`: the aspect modeling (detection) method, e.g., `-am lda`, including `rnd`, `lda`,`btm`, `ctm`, `nrl`, `bert`, `hast`, `cat`

> `-data`: the raw review file, e.g., `-data ../data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml`

> `-output`: the folder to store the pipeline outputs, e.g., `-output ../output/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml` including preprocessed reviews, trained models, predictions, evaluations, ...

`LADy` knows the methods' hyperparameters and evaluation settings from [`./src/params.py`](./src/params.py)

Here is the codebase folder structure:
```
├── src
|   ├── cmn
|   |   ├── review.py   -> class definition for review as object
|   |   ├── semeval.py  -> overridden class for semeval reviews
|   ├── aml
|   |   ├── mdl.py      -> abstract aspect model to be overridden by baselines
|   |   ├── rnd.py      -> random aspect model that randomly predicts aspects
|   |   ├── lda.py      -> unsupervised aspect detection based on LDA
|   |   ├── btm.py      -> unsupervised aspect detection based on biterm topic modeling
|   |   ├── ctm.py      -> unsupervised aspect detection based on contextual topic modeling (neural)
|   |   ├── nrl.py      -> unsupervised aspect detection based on neural topic modeling
|   ├── params.py       -> running settings of the pipeline
|   ├── main.py         -> main driver of the pipeline
```

### `-output {output}`
`LADy` runs the pipleline for `['prep', 'train', 'test', 'eval', 'agg']` steps and generates outputs in the given `-output` path:

> `['prep']`: loads raw reviews and generate review objects in `{output}/review.{list of languages}.pkl` like [`./output/toy.2016SB5/`](./output/toy.2016SB5/)
 
> `['train']`: loads review objects and create an instance of aspect modeling (detection) method given in `-am {am}`. 
> `LADy` splits reviews into `train` and `test` based on `params.settings['train']['ratio']` in [`./src/params.py`](./src/params.py).
> `LADy` further splits `train` into `params.settings['train']['nfolds']` for cross-validation and model tuning during training. 
> The result of this step is a collection of trained models for each fold in `{output}/{naspect}.{languges used for back-translation}/{am}/` like [`./output/toy.2016SB5/5.arb_Arab/lda`](./output/toy.2016SB5/5.arb_Arab/lda/)
```
├── f{k}.model            -> saved aspect model for k-th fold
├── f{k}.model.dict       -> dictionary of tokens/words for k-th fold
```

> `['test']`: predicts the aspects on the test set with `params.settings["test"]["h_ratio"] * 100` % latent aspect meaning that this percentage of the aspects will be hidden in the test reviews.
Also, the model will which has been saved in the previous step (train) will be loaded to be used for inference.
> The results of inference will be pairs of golden truth aspects with the inferred aspects sorted based on their probability that will be saved for each fold in `{output}/{naspect}/{am}/` like [`./output/toy.2016SB5/5/lda`](./output/toy.2016SB5/5/lda/)
```
├── f{k}.model.pred.{h_ratio}        -> pairs of golden truth and inferred aspects with (h_ratio * 100) % hidden aspects for k-th fold
```

> `['eval']`: evaluate the inference results in the test step and save the results for different metrics in `params.settings['eval']['metrics']` for different k in `params.settings["eval"]["topkstr"]`.
> The result of this step will be saved for each fold in `{output}/{naspect}/{am}/` like [`./output/toy.2016SB5/5/lda`](./output/toy.2016SB5/5/lda/)
```
├── f{k}.model.pred.{h_ratio}       -> evaluation of inference for k-th fold with (h_ratio * 100) % hidden aspects
├── model.pred.{h_ratio}.csv        -> mean of evaluation for all folds with (h_ratio * 100) % hidden aspects
```
 
> `['agg']`: aggregate the inferred result in this step for all the aspect models in all the folds and for all the `h_ratio` values will be saved in a file in `{output}/` like [`./output/toy.2016SB`](./output/toy.2016SB5)

```
├── agg.pred.eval.mean.csv          -> aggregated file including all inferences on a specific dataset
```

## 4. Experiment
to be completed ...
- datasets and stats
- table of results
- link to results

Due to OOV (an aspect might be in test set which is not seen in traning set during model training), we may have metric@n for n >> +inf not equal to 1.

## 5. License
©2023. This work is licensed under a [CC BY-NC-SA 4.0](LICENSE.txt) license.

Farinam Hemmatizadeh<sup>1,3</sup>, Christine Wong<sup>1, 4</sup>, Alice Yu<sup>2, 5</sup>, and [Hossein Fani](https://hosseinfani.github.io/)<sup>1,6</sup>

<sup><sup>1</sup>School of Computer Science, Faculty of Science, University of Windsor, ON, Canada.</sup>
<sup><sup>2</sup>Vincent Massey Secondary School, ON, Canada.</sup>
 <br>
<sup><sup>3</sup>[hemmatif@uwindsor.ca](mailto:hemmatif@uwindsor.ca), [f.hemmatizadeh@gmail.com](mailto:f.hemmatizadeh@gmail.com)</sup>
<sup><sup>4</sup>[wong93@uwindsor.ca](mailto:wong93@uwindsor.ca)</sup>
<sup><sup>5</sup>[qinfengyu123@gmail.com](mailto:qinfengyu123@gmail.com)</sup>
<sup><sup>6</sup>[hfani@uwindsor.ca](mailto:hfani@uwindsor.ca)</sup>

## 6. Acknowledgments
In this work, we use [`LDA`](https://radimrehurek.com/gensim/models/ldamodel.html), [`bitermplus`](https://github.com/maximtrp/bitermplus), [`OCTIS`](https://github.com/MIND-Lab/OCTIS), [`pytrec_eval`](https://github.com/cvangysel/pytrec_eval), [`SimAlign`](https://github.com/cisnlp/simalign), [`DeCLUTR`](https://github.com/JohnGiorgi/DeCLUTR), [`No Language Left Behind (NLLB)`](https://github.com/facebookresearch/fairseq/tree/nllb), and other libraries and models. We extend our gratitude to the respective authors of these resources for their valuable contributions.


## 7. Awards

> [`CAD$150, Silver Medalist, UWill Discover 2023`](https://www.uwindsor.ca/uwilldiscover/312/uwill-discover-awards) 👉 [`slides`](./misc/UWillDiscover23.pdf)
> <p align="center"><img src='./misc/cs_demo_day_23april23.png' width="350" ></p>
> <p align="center">From Left: Soroush, Atefeh, Christine, Farinam, Mohammad</p>


## 8. Contributing
We strongly encourage and welcome pull requests from contributors. If you plan to make substantial modifications, we kindly request that you first open an issue to initiate a discussion. This will allow us to have a clear understanding of the modifications you intend to make and ensure a smooth collaboration process.
