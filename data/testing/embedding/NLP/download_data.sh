#!/bin/bash

NLP=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$NLP" || exit 1
ROOT="$(readlink -f .)"
CLS="$ROOT/classification"
REG="$ROOT/regression"
mkdir -p "$CLS" "$REG"

module load git-lfs/2.13.2
git lfs install


## Classification
cd "$CLS" || exit 1
git clone git@hf.co:datasets/dair-ai/emotion
git clone git@hf.co:datasets/Yelp/yelp_review_full
git clone git@hf.co:datasets/fancyzhx/ag_news
git clone git@hf.co:datasets/lmsys/toxic-chat
git clone git@hf.co:datasets/cornell-movie-review-data/rotten_tomatoes
git clone git@hf.co:datasets/google-research-datasets/go_emotions
git clone git@hf.co:datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0
git clone git@hf.co:datasets/cardiffnlp/tweet_topic_single
git clone git@hf.co:datasets/google-research-datasets/poem_sentiment
git clone git@hf.co:datasets/tyqiangz/multilingual-sentiments
git clone git@hf.co:datasets/cardiffnlp/tweet_sentiment_multilingual
git clone git@hf.co:datasets/Sp1786/multiclass-sentiment-analysis-dataset
git clone git@hf.co:datasets/OxAISH-AL-LLM/wiki_toxic
git clone git@hf.co:datasets/Arsive/toxicity_classification_jigsaw
git clone git@hf.co:datasets/sealuzh/app_reviews
git clone git@hf.co:datasets/nickmuchi/financial-classification
git clone git@hf.co:datasets/climatebert/climate_sentiment
git clone git@hf.co:datasets/RuyuanWan/Dynasent_Disagreement

## Regression
cd "$REG" || exit 1
git clone git@hf.co:datasets/agentlans/readability
git clone git@hf.co:datasets/jhan21/amazon-food-reviews-dataset
git clone git@hf.co:datasets/OsamaBsher/AITA-Reddit-Dataset
git clone git@hf.co:datasets/turkish-nlp-suite/vitamins-supplements-reviews
git clone git@hf.co:datasets/mofosyne/stupidfilter