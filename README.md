---
language: "en"
tags:
- sentiment
---


# Overview
This model is a fine-tuned checkpoint of [RoBERTa-large](https://huggingface.co/roberta-large) (Liu et al. 2019). It enables reliable binary sentiment analysis for various types of English-language text. For each instance it predicts either positive (1) or negative (0) sentiment. The model was fine-tuned and evaluated on 15 data sets from diverse text sources to enhance generalization across different types of texts (reviews, tweets, etc.). Consequently, it outperforms models trained on only one type of text (e.g., movie reviews from the popular SST-2 benchmark) when used on new data as shown below. 
 
# Usage
The model can be used with few lines of code as shown below. We suggest that you manually label a subset of your data to evaluate performance for your use case. For performance benchmark values across different sentiment analysis contexts, refer to our paper ([Heitmann et al. 2020](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489963)). The model can also be used as a starting point for further [fine-tuning](https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-trainer) on your sentiment analysis task.

The easiest way to use the model employs Huggingface's [sentiment analysis pipeline](https://huggingface.co/transformers/quicktour.html#getting-started-on-a-task-with-a-pipeline):
```
from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english")
print(sentiment_analysis("I love this!"))
```

Alternatively you can load the model as follows:
``` 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
```


 
# Performance
To evaluate the performance of our general-purpose sentiment analysis model, we set aside an evaluation set from each data set, which was not used for training. On average, our model outperforms a [DistilBERT-based model](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) (which is solely fine-tuned on the popular SST-2 data set) by more than 14 percentage points (78.1 vs. 92.2, see table below). As a robustness check, we evaluate the model in a leave-on-out manner (training on 14 data sets, evaluating on the one left out), which decreases model performance by only about 3 percentage points on average and underscores its generalizability.

|Dataset|DistilBERT SST-2|This model|
|---|---|---|
|McAuley and Leskovec (2013) (Reviews)|84.7|95.5|
|McAuley and Leskovec (2013) (Review Titles)|65.5|87.5|
|Yelp Academic Dataset|84.8|97.0|
|Maas et al. (2011)|80.6|91.5|
|Kaggle|87.2|96.0|
|Pang and Lee (2005)|89.7|91.0|
|Nakov et al. (2013)|70.1|89.5|
|Shamma (2009)|76.0|90.0|
|Blitzer et al. (2007) (Books)|83.0|93.5|
|Blitzer et al. (2007) (DVDs)|84.5|90.5|
|Blitzer et al. (2007) (Electronics)|74.5|95.0|
|Blitzer et al. (2007) (Kitchen devices)|80.0|97.5|
|Pang et al. (2002)|73.5|83.0|
|Speriosu et al. (2011)|71.5|88.0|
|Hartmann et al. (2019)|65.5|98.0|
|**Average**|**78.1**|**92.2**|
 
# Fine-tuning hyperparameters
- learning_rate = 2e-5
- batch_size = 8
- max_seq_length = 128
- num_train_epochs = 3.0
  
# Citation
Please cite [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3489963) when you use our model.

```
@article{heitmann2020,
  title={More than a feeling: Benchmarks for sentiment analysis accuracy},
  author={Heitmann, Mark and Siebert, Christian and Hartmann, Jochen and Schamp, Christina},
  journal={Available at SSRN 3489963},
  year={2020}
}
```
