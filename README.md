---
language: "en"
tags:
- sentiment
- twitter
- reviews
---


# Overview
This model is a fine-tuned checkpoint of [RoBERTa-large](https://huggingface.co/roberta-large) ([Liu et al. 2019](https://arxiv.org/pdf/1907.11692.pdf)). It enables reliable binary sentiment analysis for various types of English-language text. For each instance, it predicts either positive (1) or negative (0) sentiment. The model was fine-tuned and evaluated on 15 data sets from diverse text sources to enhance generalization across different types of texts (reviews, tweets, etc.). Consequently, it outperforms models trained on only one type of text (e.g., movie reviews from the popular SST-2 benchmark) when used on new data as shown below. 


# Predictions on a data set
If you want to predict sentiment for your own data, we provide an example script via [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb). You can load your data to a Google Drive and run the script for free on a Colab GPU. Set-up only takes a few minutes. We suggest that you manually label a subset of your data to evaluate performance for your use case. For performance benchmark values across various sentiment analysis contexts, please refer to our paper ([Heitmann et al. 2020](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489963)).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chrsiebert/sentiment-roberta-large-english/blob/main/sentiment_roberta_prediction_example.ipynb)


# Use in a Hugging Face pipeline
The easiest way to use the model for single predictions is Hugging Face's [sentiment analysis pipeline](https://huggingface.co/transformers/quicktour.html#getting-started-on-a-task-with-a-pipeline), which only needs a couple lines of code as shown in the following example:
```
from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
print(sentiment_analysis("I love this!"))
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chrsiebert/sentiment-roberta-large-english/blob/main/sentiment_roberta_pipeline.ipynb)


# Use for further fine-tuning
The model can also be used as a starting point for further fine-tuning of RoBERTa on your specific data. Please refer to Hugging Face's [documentation](https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-trainer) for further details and example code.


# Performance
To evaluate the performance of our general-purpose sentiment analysis model, we set aside an evaluation set from each data set, which was not used for training. On average, our model outperforms a [DistilBERT-based model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) (which is solely fine-tuned on the popular SST-2 data set) by more than 15 percentage points (78.1 vs. 93.2 percent, see table below). As a robustness check, we evaluate the model in a leave-one-out manner (training on 14 data sets, evaluating on the one left out), which decreases model performance by only about 3 percentage points on average and underscores its generalizability. Model performance is given as evaluation set accuracy in percent.

|Dataset|DistilBERT SST-2|This model|
|---|---|---|
|McAuley and Leskovec (2013) (Reviews)|84.7|98.0|
|McAuley and Leskovec (2013) (Review Titles)|65.5|87.0|
|Yelp Academic Dataset|84.8|96.5|
|Maas et al. (2011)|80.6|96.0|
|Kaggle|87.2|96.0|
|Pang and Lee (2005)|89.7|91.0|
|Nakov et al. (2013)|70.1|88.5|
|Shamma (2009)|76.0|87.0|
|Blitzer et al. (2007) (Books)|83.0|92.5|
|Blitzer et al. (2007) (DVDs)|84.5|92.5|
|Blitzer et al. (2007) (Electronics)|74.5|95.0|
|Blitzer et al. (2007) (Kitchen devices)|80.0|98.5|
|Pang et al. (2002)|73.5|95.5|
|Speriosu et al. (2011)|71.5|85.5|
|Hartmann et al. (2019)|65.5|98.0|
|**Average**|**78.1**|**93.2**|
 
# Fine-tuning hyperparameters
- learning_rate = 2e-5
- num_train_epochs = 3.0
- warmump_steps = 500
- weight_decay = 0.01

Other values were left at their defaults as listed [here](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments).
  
# Citation and contact
Please cite [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489963) when you use our model. Feel free to reach out to [christian.siebert@uni-hamburg.de](mailto:christian.siebert@uni-hamburg.de) with any questions or feedback you may have.

```
@article{heitmann2020,
  title={More than a feeling: Benchmarks for sentiment analysis accuracy},
  author={Heitmann, Mark and Siebert, Christian and Hartmann, Jochen and Schamp, Christina},
  journal={Available at SSRN 3489963},
  year={2020}
}
```
