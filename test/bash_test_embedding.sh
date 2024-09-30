#!/bin/bash

TESTS=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT="$(dirname "$TESTS")"
TEST_DATA="$ROOT/data/testing/embedding"
TEMPTEST="$ROOT/bash_manual_test_temp"
rm -rf "$TEMPTEST"

if [[ -z "${CC_CLUSTER}" ]]; then
    echo "On local machine, will use virtual environment for testing"
    PYTHON="$ROOT/.venv/bin/python"
else
    # shellcheck disable=SC2016
    echo 'On Compute Canada, will use container-defined $PYTHON variable'
    module load apptainer
    export APPTAINERENV_MPLCONFIGDIR="$(readlink -f .)"/.mplconfig
    export APPTAINERENV_OPENBLAS_NUM_THREADS="1"
    apptainer run --home "$(readlink -f .)" df_analyze.sif "$(readlink -f test/cc_bash_test_embedding.sh)"
    exit 0
fi


declare -a NLP_DATASETS=(
  "$TEST_DATA/NLP/classification/ag_news/all.parquet"
  "$TEST_DATA/NLP/classification/climate_sentiment/all.parquet"
  "$TEST_DATA/NLP/classification/Dynasent_Disagreement/all.parquet"
  "$TEST_DATA/NLP/classification/financial-classification/all.parquet"
  "$TEST_DATA/NLP/classification/multiclass-sentiment-analysis-dataset/all.parquet"
  "$TEST_DATA/NLP/classification/poem_sentiment/all.parquet"
  "$TEST_DATA/NLP/classification/rotten_tomatoes/all.parquet"
  "$TEST_DATA/NLP/classification/toxicity_classification_jigsaw/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_sentiment_multilingual/data/arabic/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_sentiment_multilingual/data/english/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_sentiment_multilingual/data/french/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_sentiment_multilingual/data/german/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_sentiment_multilingual/data/hindi/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_sentiment_multilingual/data/italian/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_sentiment_multilingual/data/portuguese/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_sentiment_multilingual/data/spanish/all.parquet"
  "$TEST_DATA/NLP/classification/tweet_topic_single/all_2020.parquet"
  "$TEST_DATA/NLP/classification/tweet_topic_single/all_2021.parquet"
  "$TEST_DATA/NLP/classification/wiki_toxic/all.parquet"
  "$TEST_DATA/NLP/classification/yelp_review_full/all.parquet"
  "$TEST_DATA/NLP/regression/AITA-Reddit-Dataset/all.parquet"
  "$TEST_DATA/NLP/regression/amazon-food-reviews-dataset/all.parquet"
  "$TEST_DATA/NLP/regression/readability/all_arxiv.parquet"
  "$TEST_DATA/NLP/regression/readability/all_fineweb.parquet"
  "$TEST_DATA/NLP/regression/readability/all_tinystories.parquet"
  "$TEST_DATA/NLP/regression/readability/all_wikipedia.parquet"
  "$TEST_DATA/NLP/regression/stupidfilter/all.parquet"
  "$TEST_DATA/NLP/regression/vitamins-supplements-reviews/all.parquet"
  # "$TEST_DATA/NLP/classification/go_emotions/all.parquet"  # multilabel
)

declare -a VISION_DATASETS=(
  "$TEST_DATA/vision/classification/Anime-dataset/all.parquet"
  "$TEST_DATA/vision/classification/Brain-Tumor-Classification/all.parquet"
  "$TEST_DATA/vision/classification/GenAI-Bench-1600/all.parquet"
  "$TEST_DATA/vision/classification/Handwritten-Mathematical-Expression-Convert-LaTeX/all.parquet"
  "$TEST_DATA/vision/classification/brain-mri-dataset/all.parquet"
  "$TEST_DATA/vision/classification/data-food-classification/all.parquet"
  "$TEST_DATA/vision/classification/fast_food_image_classification/all.parquet"
  "$TEST_DATA/vision/classification/garbage_detection/all.parquet"
  "$TEST_DATA/vision/classification/nsfw_detect/all.parquet"
  "$TEST_DATA/vision/regression/body-measurements-dataset/all.parquet"
  # "$TEST_DATA/vision/classification/Places_in_Japan/all.parquet"  # missing labels
  # "$TEST_DATA/vision/classification/Visual_Emotional_Analysis/all.parquet"  # missing labels
  # "$TEST_DATA/vision/classification/rare-species/all.parquet"  # text labels
)
N_NLP=${#NLP_DATASETS[@]}
N_VISION=${#VISION_DATASETS[@]}

declare -a NLP_OUTDIRS=(
  "$TEMPTEST/NLP/classification/ag_news"
  "$TEMPTEST/NLP/classification/climate_sentiment"
  "$TEMPTEST/NLP/classification/Dynasent_Disagreement"
  "$TEMPTEST/NLP/classification/financial-classification"
  "$TEMPTEST/NLP/classification/multiclass-sentiment-analysis-dataset"
  "$TEMPTEST/NLP/classification/poem_sentiment"
  "$TEMPTEST/NLP/classification/rotten_tomatoes"
  "$TEMPTEST/NLP/classification/toxicity_classification_jigsaw"
  "$TEMPTEST/NLP/classification/tweet_sentiment_multilingual/data/arabic"
  "$TEMPTEST/NLP/classification/tweet_sentiment_multilingual/data/english"
  "$TEMPTEST/NLP/classification/tweet_sentiment_multilingual/data/french"
  "$TEMPTEST/NLP/classification/tweet_sentiment_multilingual/data/german"
  "$TEMPTEST/NLP/classification/tweet_sentiment_multilingual/data/hindi"
  "$TEMPTEST/NLP/classification/tweet_sentiment_multilingual/data/italian"
  "$TEMPTEST/NLP/classification/tweet_sentiment_multilingual/data/portuguese"
  "$TEMPTEST/NLP/classification/tweet_sentiment_multilingual/data/spanish"
  "$TEMPTEST/NLP/classification/tweet_topic_single"
  "$TEMPTEST/NLP/classification/tweet_topic_single"
  "$TEMPTEST/NLP/classification/wiki_toxic"
  "$TEMPTEST/NLP/classification/yelp_review_full"
  "$TEMPTEST/NLP/regression/AITA-Reddit-Dataset"
  "$TEMPTEST/NLP/regression/amazon-food-reviews-dataset"
  "$TEMPTEST/NLP/regression/readability/arxiv"
  "$TEMPTEST/NLP/regression/readability/fineweb"
  "$TEMPTEST/NLP/regression/readability/tinystories"
  "$TEMPTEST/NLP/regression/readability/wikipedia"
  "$TEMPTEST/NLP/regression/stupidfilter"
  "$TEMPTEST/NLP/regression/vitamins-supplements-reviews"
  # "$TEMPTEST/NLP/classification/go_emotions/"  # multilabel
)

declare -a VISION_OUTDIRS=(
  "$TEMPTEST/vision/classification/Anime-dataset"
  "$TEMPTEST/vision/classification/Brain-Tumor-Classification"
  "$TEMPTEST/vision/classification/GenAI-Bench-1600"
  "$TEMPTEST/vision/classification/Handwritten-Mathematical-Expression-Convert-LaTeX"
  "$TEMPTEST/vision/classification/brain-mri-dataset"
  "$TEMPTEST/vision/classification/data-food-classification"
  "$TEMPTEST/vision/classification/fast_food_image_classification"
  "$TEMPTEST/vision/classification/garbage_detection"
  "$TEMPTEST/vision/classification/nsfw_detect"
  "$TEMPTEST/vision/regression/body-measurements-dataset"
  # "$TEMPTEST/vision/classification/Places_in_Japan"  # missing labels
  # "$TEMPTEST/vision/classification/Visual_Emotional_Analysis"  # missing labels
  # "$TEMPTEST/vision/classification/rare-species/"  # text labels
)

# echo "$ROOT"
cd "$ROOT" || exit 1

echo "Ensuring NLP Dataset 'all.parquet' files exist..."
"$PYTHON" "$ROOT/src/df_analyze/embedding/_generate_all_parquets.py"

echo "================================================================================="
echo "                              Testing NLP Datasets                             "
echo "================================================================================="

for (( i=0; i<${N_NLP}; i++ ));
do
    outdir="${NLP_OUTDIRS[$i]}"
    out="$outdir/out.parquet"
    name="$(basename "$outdir")"
    echo -n "Testing $name: "
    mkdir -p "$outdir"
    "$PYTHON" df-embed.py --data "${NLP_DATASETS[$i]}" --modality nlp --out "$out" --name "$name" --limit-samples 8
    sleep 2  # lol
    if [ ! -f "$out" ]; then
        echo "Failed to produce embedded file: $out"
        rm -rf "$TEMPTEST"  || echo "Failed to remove tempdir $TEMPTEST. Remove manually"
        exit 1
    else
        rm -f "$out"  || echo "Failed to remove tempfile $out. Remove manually"
        rm -rf "$outdir"  || echo "Failed to remove tempdir $outdir. Remove manually"
    fi
done

echo "================================================================================="
echo "                            Testing Vision Datasets                            "
echo "================================================================================="

for (( i=0; i<${N_VISION}; i++ ));
do
    outdir="${VISION_OUTDIRS[$i]}"
    out="$outdir/out.parquet"
    name="$(basename "$outdir")"
    echo -n "Testing $name: "
    mkdir -p "$outdir"
    "$PYTHON" df-embed.py --data "${VISION_DATASETS[$i]}" --modality vision --out "$out" --name "$name" --limit-samples 8
    sleep 2  # lol
    if [ ! -f "$out" ]; then
        echo "Failed to produce embedded file: $out"
        rm -rf "$TEMPTEST"  || echo "Failed to remove tempdir $TEMPTEST. Remove manually"
        exit 1
    else
        rm -f "$out"  || echo "Failed to remove tempfile $out. Remove manually"
        rm -rf "$outdir"  || echo "Failed to remove tempdir $outdir. Remove manually"
    fi
done

rm -rf "$TEMPTEST"
