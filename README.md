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
- [7. Contribution](#7-contribution)

</td>
<td>
  <p align="center">
 <img src='./src/cmn/LADy.png' width="550" >
<!--  <br> -->
<!--  <a href="https://lucid.app/lucidchart/fe256064-3fda-465a-9abc-036dfc40acad/edit?view_items=svRVuxyZvY9n%2CsvRVVLD91NpJ%2CxDRV-pti53Ae%2CwJRVh7la6C-y%2CBLRV4aXmE.uY%2CBLRVOyM~DMFW&invitationId=inv_6e8aa9a6-1854-4ecf-a753-e1b2e05b50fc">class diagram for review</a> -->
</p>
</td>
</tr>
</table>


## 1. Setup
`LADy` has been developed on `Python 3.8` and can be installed by `conda` or `pip`, `docker`:

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

```bash
docker run -it --name lady_container ghcr.io/fani-lab/lady:main
```
This command installs compatible versions of the following libraries:

> [`./src/cmn`](./src/cmn): `transformers, sentence_transformers, scipy, simalign, nltk`

> [`./src/aml`](./src/aml): `gensim, nltk, pandas, requests, bitermplus, contextualized_topic_models`

> others: `pytrec-eval-terrier, sklearn, matplotlib, seaborn, tqdm`

The following aspect detection baselines will be also cloned as submodules:
> [`bert-e2e-absa`](https://aclanthology.org/D19-5505/) → [`./src/bert-e2e-absa`](https://github.com/fani-lab/BERT-E2E-ABSA)

> [`hast`](https://www.ijcai.org/proceedings/2018/0583) → [`./src/hast`](https://github.com/fani-lab/HAST)

> [`cat`](https://aclanthology.org/2020.acl-main.290/) → [`./src/cat`](https://github.com/fani-lab/cat)

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
Pickle file is later used by models for training and testing purposes. Sample pickle files for a `toy` dataset: [`./output/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml`](./output/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml), there are some
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

> [`Local LDA [Brody and Elhadad, NAACL2010]`](https://aclanthology.org/N10-1122/) in [`./src/aml/lda.py`](./src/aml/lda.py),

> [`Biterm Topic Modeling [WWW2013]`](https://dl.acm.org/doi/10.1145/2488388.2488514) in [`./src/aml/btm.py`](./src/aml/btm.py),

> [`Contextual Topic Modeling [EACL2021]`](https://aclanthology.org/2021.eacl-main.143/) in [`./src/aml/ctm.py`](./src/aml/ctm.py),

> [`BERT-E2E-ABSA [W-NUT@EMNLP2019]`](https://aclanthology.org/D19-5505/) in [`./src/bert-e2e-absa`](https://github.com/fani-lab/BERT-E2E-ABSA)
 
> [`HAST [IJCAI2018]`](https://aclanthology.org/2021.eacl-main.143/) in [`./src/hast`](https://github.com/fani-lab/HAST)
 
> [`CAt [ACL2020]`](https://aclanthology.org/2020.acl-main.290/) in [`./src/cat`](https://github.com/fani-lab/cat),

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

> `['eval']`: evaluate the inference results in the test step and save the results for different metrics in [`params.settings['eval']['metrics']`](https://github.com/fani-lab/LADy/blob/main/src/params.py#L55) for different k in `params.settings["eval"]["topkstr"]`.
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

We conducted a series of experiments involving backtranslation using `six` different `natural languages` that belong to `diverse language families`. These experiments aimed to explore the effect of `backtranslation augmentation` across various aspect detection methods and domains, particularly in the context of `restaurant` and `laptop` `reviews`, where aspects may not necessarily be explicitly mentioned but are implicitly present with no surface form (`latent`). Through our findings, we observed a synergistic impact, indicating that the utilization of `backtranslation` enhances the performance of `aspect detection` whether the aspect is 'explicit' or 'latent'.

### Datasets

`LADy` utilizes state-of-the-art `semeval` datasets to `augment` the english datasets with `backtranslation` via different languages and evaluate `latent aspect detection`. Specifically, training sets from `semeval-14` for restaurant and laptop reviews, as well as restaurant reviews from `semeval-15` and `semeval-16` are employed. Moreover, we have created a compact and simplified version of the original datasets, referred to as a `toy dataset`, for our experimental purposes.

| dataset               | file (.xml)                                                                                                                                                                          |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| semeval-14-laptop     | [`./data/raw/semeval/SemEval-14/Laptop_Train_v2.xml`](./data/raw/semeval/SemEval-14/Laptop_Train_v2.xml)                                                                             |
| semeval-14-restaurant | [`./data/raw/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml`](./data/raw/semeval/SemEval-14/Semeval-14-Restaurants_Train.xml)                                                   |
| semeval-15-restaurant | [`./data/raw/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml`](./data/raw/semeval/2015SB12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml) |
| semeval-16-restaurant | [`./data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml`](./data/raw/semeval/2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml)                                                   |
| toy                   | [`./data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml`](./data/raw/semeval/toy.2016SB5/ABSA16_Restaurants_Train_SB1_v2.xml)                                           |

### Statistics on original and backtranslated reviews

The reviews were divided into sentences, and our experiments were conducted on each sentence treated as an individual review, assuming that each sentence represents a single aspect. The statistics of the datasets can be seen in the table below.

|                       |                              |              | exact match |        |        |        |        |         |
|-----------------------|------------------------------|--------------|-------------|--------|--------|--------|--------|---------|
| dataset               | #reviews  | avg #aspects | chinese  | farsi  | arabic | french | german | spanish |
| semeval-14-laptop     | 1,488     | 1.5846       | 0.1763   | 0.2178 | 0.2727 | 0.3309 | 0.3214 | 0.3702  |
| semeval-14-restaurant | 2,023     | 1.8284       | 0.1831   | 0.2236 | 0.2929 | 0.3645 | 0.3724 | 0.4088  |
| semeval-15-restaurant | 0,833     | 1.5354       | 0.2034   | 0.2312 | 0.3021 | 0.3587 | 0.3907 | 0.4128  |
| semeval-16-restaurant | 1,234     | 1.5235       | 0.2023   | 0.2331 | 0.2991 | 0.3556 | 0.3834 | 0.4034  |

### Results

The average performances of 5-fold models with backtranslation and lack thereof have been reported in our [`CIKM23 paper`](https://hosseinfani.github.io/res/papers/2023_CIKM_Latent_Aspect_Detection_via_Backtranslation_Augmentation.pdf), also shown below:

![image](https://github.com/fani-lab/LADy/assets/8619934/af8239cd-b536-4171-b956-6b9f82315bac)

The table below presents the provided links to directories that hold the remaining results of our experiment. These directories consist of diverse `aspect detection` models applied to different `datasets` and `languages`, with varying percentages of `latent` aspects.

| dataset               | review files (english, chinese, farsi, arabic, french, german, spanish, and all) and results' directory                                                                                                                                                                                              |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| semeval-14-laptop     | [`./output/Semeval-14/Laptop/`]([./output/Semeval-14/Laptop/](https://uwin365.sharepoint.com/sites/cshfrg-ReviewAnalysis/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fcshfrg%2DReviewAnalysis%2FShared%20Documents%2FLADy%2FLADy0%2E1%2E0%2E0%2Foutput%2FSemEval%2D14%2FLaptop&viewid=4cd69493%2D951c%2D47b5%2Db34a%2Dc1cdbf3a0412))                                                                                                                                                                                                                                         22.0 MB |
| semeval-14-restaurant | [`./output/Semeval-14/Restaurants/`]([./output/Semeval-14/Restaurants/](https://uwin365.sharepoint.com/sites/cshfrg-ReviewAnalysis/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fcshfrg%2DReviewAnalysis%2FShared%20Documents%2FLADy%2FLADy0%2E1%2E0%2E0%2Foutput%2FSemEval%2D14%2FRestaurants&viewid=4cd69493%2D951c%2D47b5%2Db34a%2Dc1cdbf3a0412))                                                                                                                                                                                                                                    22.2 MB |
| semeval-15-restaurant | [`./output/2015SB12/`](https://uwin365.sharepoint.com/sites/cshfrg-ReviewAnalysis/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fcshfrg%2DReviewAnalysis%2FShared%20Documents%2FLADy%2FLADy0%2E1%2E0%2E0%2Foutput%2F2015SB12&viewid=4cd69493%2D951c%2D47b5%2Db34a%2Dc1cdbf3a0412)    53.1 GB  |
| semeval-16-restaurant | [`./output/2016SB5/`](https://uwin365.sharepoint.com/sites/cshfrg-ReviewAnalysis/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fcshfrg%2DReviewAnalysis%2FShared%20Documents%2FLADy%2FLADy0%2E1%2E0%2E0%2Foutput%2F2016SB5&viewid=4cd69493%2D951c%2D47b5%2Db34a%2Dc1cdbf3a0412)    103 MB  |
| toy                   | [`./output/toy.2016SB5/`](https://uwin365.sharepoint.com/sites/cshfrg-ReviewAnalysis/Shared%20Documents/Forms/AllItems.aspx?ga=1&id=%2Fsites%2Fcshfrg%2DReviewAnalysis%2FShared%20Documents%2FLADy%2FLADy0%2E1%2E0%2E0%2Foutput%2Ftoy%2E2016SB5&viewid=4cd69493%2D951c%2D47b5%2Db34a%2Dc1cdbf3a0412) 64.6 MB |

Due to OOV (an aspect might be in test set which is not seen in traning set during model training), we may have metric@n for n >> +inf not equal to 1.

## 5. License
©2024. This work is licensed under a [CC BY-NC-SA 4.0](LICENSE.txt) license.

## 6. Acknowledgments
In this work, we use [`LDA`](https://radimrehurek.com/gensim/models/ldamodel.html), [`bitermplus`](https://github.com/maximtrp/bitermplus), [`OCTIS`](https://github.com/MIND-Lab/OCTIS), [`pytrec_eval`](https://github.com/cvangysel/pytrec_eval), [`SimAlign`](https://github.com/cisnlp/simalign), [`DeCLUTR`](https://github.com/JohnGiorgi/DeCLUTR), [`No Language Left Behind (NLLB)`](https://github.com/facebookresearch/fairseq/tree/nllb), and other libraries and models. We extend our gratitude to the respective authors of these resources for their valuable contributions.



## 7. Contributing
We strongly encourage and welcome pull requests from contributors. If you plan to make substantial modifications, we kindly request that you first open an issue to initiate a discussion. This will allow us to have a clear understanding of the modifications you intend to make and ensure a smooth collaboration process.
