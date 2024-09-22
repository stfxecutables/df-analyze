#!/bin/bash

# bghira/pseudo-camera-10k                               # 5.2GB, 2 800 images
# BaiqiL/GenAI-Bench-1600                                # 4.8GB, 9 600 images
# imageomics/rare-species                                # 4.3GB, 11,983 images
# deepghs/nsfw_detect                                    # 1.8GB, 28 000 images
#
# JapanDegitalMaterial/Places_in_Japan                   # 504MB, 149 images
# Borismile/Anime-dataset                                # 291MB, 1 882 images
#
# sartajbhuvaji/Brain-Tumor-Classification               #  91MB, 3264 images
# Kaludi/data-food-classification                        #  71MB, 1400 images
# Azu/Handwritten-Mathematical-Expression-Convert-LaTeX  #  46MB, 12,167
# andrewsunanda/fast_food_image_classification           #  40MB, 3000 images
# zz990906/garbage_detection                             #  12MB, 640 images
# FastJobs/Visual_Emotional_Analysis                     #  11MB, 800 images
# TrainingDataPro/brain-mri-dataset                      #  10MB, 275 images
#
# # regression?
# TrainingDataPro/body-measurements-dataset              # 43.5MB, 315 images

# 39G     classification/pseudo-camera-10k
# 9.7G    classification/rare-species
# 6.6G    classification/GenAI-Bench-1600
# 3.4G    classification/nsfw_detect
# 857M    classification/Anime-dataset
# 746M    classification/fast_food_image_classification
# 669M    classification/brain-mri-dataset
# 490M    classification/data-food-classification
# 202M    classification/Visual_Emotional_Analysis
# 174M    classification/Brain-Tumor-Classification
# 162M    classification/garbage_detection
# 51M     classification/Handwritten-Mathematical-Expression-Convert-LaTeX
# 91M     classification/Places_in_Japan
# 126M    regression/body-measurements-dataset

NLP=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$NLP" || exit 1
ROOT="$(readlink -f .)"
CLS="$ROOT/classification"
REG="$ROOT/regression"
mkdir -p "$CLS" "$REG"

module load git-lfs/2.13.2
git lfs install

# ## Classification
# cd "$CLS" || exit 1

# git clone git@hf.co:datasets/bghira/pseudo-camera-10k
# git clone git@hf.co:datasets/BaiqiL/GenAI-Bench-1600
# git clone git@hf.co:datasets/imageomics/rare-species
# git clone git@hf.co:datasets/deepghs/nsfw_detect

# git clone git@hf.co:datasets/JapanDegitalMaterial/Places_in_Japan
# git clone git@hf.co:datasets/Borismile/Anime-dataset

# git clone git@hf.co:datasets/sartajbhuvaji/Brain-Tumor-Classification
# git clone git@hf.co:datasets/Kaludi/data-food-classification
# git clone git@hf.co:datasets/Azu/Handwritten-Mathematical-Expression-Convert-LaTeX
# git clone git@hf.co:datasets/andrewsunanda/fast_food_image_classification
# git clone git@hf.co:datasets/zz990906/garbage_detection
# git clone git@hf.co:datasets/FastJobs/Visual_Emotional_Analysis
# git clone git@hf.co:datasets/TrainingDataPro/brain-mri-dataset

# ## Regression
# cd "$REG" || exit 1
# git clone git@hf.co:datasets/TrainingDataPro/body-measurements-dataset

# curl -X GET 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/deepghs/nsfw_detect/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/JapanDegitalMaterial/Places_in_Japan/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/Borismile/Anime-dataset/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/sartajbhuvaji/Brain-Tumor-Classification/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/Kaludi/data-food-classification/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/Azu/Handwritten-Mathematical-Expression-Convert-LaTeX/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/andrewsunanda/fast_food_image_classification/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/zz990906/garbage_detection/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/FastJobs/Visual_Emotional_Analysis/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/TrainingDataPro/brain-mri-dataset/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/TrainingDataPro/brain-mri-dataset/parquet/default'
# printf "\n\n"
# curl -X GET 'https://huggingface.co/api/datasets/TrainingDataPro/body-measurements-dataset/parquet/default'
# printf "\n\n"

# ## Classification
cd "$CLS" || exit 1

# mkdir -p "$CLS/pseudo-camera-10k" || exit 1  # no labels!
mkdir -p "$CLS/GenAI-Bench-1600" || exit 1
mkdir -p "$CLS/rare-species" || exit 1
mkdir -p "$CLS/nsfw_detect" || exit 1
mkdir -p "$CLS/Places_in_Japan" || exit 1
mkdir -p "$CLS/Anime-dataset" || exit 1
mkdir -p "$CLS/Brain-Tumor-Classification" || exit 1
mkdir -p "$CLS/data-food-classification" || exit 1
mkdir -p "$CLS/Handwritten-Mathematical-Expression-Convert-LaTeX" || exit 1
mkdir -p "$CLS/fast_food_image_classification" || exit 1
mkdir -p "$CLS/garbage_detection" || exit 1
mkdir -p "$CLS/Visual_Emotional_Analysis" || exit 1
mkdir -p "$CLS/brain-mri-dataset" || exit 1
mkdir -p "$REG/body-measurements-dataset" || exit 1

# cd "$CLS/pseudo-camera-10k" || exit 1
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/0.parquet'
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/1.parquet'
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/2.parquet'
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/3.parquet'
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/4.parquet'
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/5.parquet'
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/6.parquet'
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/7.parquet'
# wget 'https://huggingface.co/api/datasets/bghira/pseudo-camera-10k/parquet/default/train/8.parquet'


# cd "$CLS/GenAI-Bench-1600" || exit 1
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/0.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/1.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/2.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/3.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/4.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/5.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/6.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/7.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/8.parquet'
# wget 'https://huggingface.co/api/datasets/BaiqiL/GenAI-Bench-1600/parquet/default/train/9.parquet'

cd "$CLS/rare-species" || exit 1
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/0.parquet'
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/1.parquet'
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/2.parquet'
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/3.parquet'
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/4.parquet'
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/5.parquet'
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/6.parquet'
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/7.parquet'
wget 'https://huggingface.co/api/datasets/imageomics/rare-species/parquet/default/train/8.parquet'

cd "$CLS/nsfw_detect" || exit 1
wget 'https://huggingface.co/api/datasets/deepghs/nsfw_detect/parquet/default/train/0.parquet'
wget 'https://huggingface.co/api/datasets/deepghs/nsfw_detect/parquet/default/train/1.parquet'
wget 'https://huggingface.co/api/datasets/deepghs/nsfw_detect/parquet/default/train/2.parquet'
wget 'https://huggingface.co/api/datasets/deepghs/nsfw_detect/parquet/default/train/3.parquet'

cd "$CLS/Places_in_Japan" || exit 1
wget 'https://huggingface.co/api/datasets/JapanDegitalMaterial/Places_in_Japan/parquet/default/train/0.parquet'

cd "$CLS/Anime-dataset" || exit 1
wget 'https://huggingface.co/api/datasets/Borismile/Anime-dataset/parquet/default/train/0.parquet'

cd "$CLS/Brain-Tumor-Classification" || exit 1
wget 'https://huggingface.co/api/datasets/sartajbhuvaji/Brain-Tumor-Classification/parquet/default/Testing/0.parquet'

cd "$CLS/data-food-classification" || exit 1
wget 'https://huggingface.co/api/datasets/Kaludi/data-food-classification/parquet/default/train/0.parquet'
wget 'https://huggingface.co/api/datasets/Kaludi/data-food-classification/parquet/default/validation/0.parquet'

cd "$CLS/Handwritten-Mathematical-Expression-Convert-LaTeX" || exit 1
wget 'https://huggingface.co/api/datasets/Azu/Handwritten-Mathematical-Expression-Convert-LaTeX/parquet/default/train/0.parquet'

cd "$CLS/fast_food_image_classification" || exit 1
wget 'https://huggingface.co/api/datasets/andrewsunanda/fast_food_image_classification/parquet/default/train/0.parquet'

cd "$CLS/garbage_detection" || exit 1
wget 'https://huggingface.co/api/datasets/zz990906/garbage_detection/parquet/default/test/0.parquet'
wget 'https://huggingface.co/api/datasets/zz990906/garbage_detection/parquet/default/train/0.parquet'
wget 'https://huggingface.co/api/datasets/zz990906/garbage_detection/parquet/default/validation/0.parquet'

cd "$CLS/Visual_Emotional_Analysis" || exit 1
wget 'https://huggingface.co/api/datasets/FastJobs/Visual_Emotional_Analysis/parquet/default/train/0.parquet'

cd "$CLS/brain-mri-dataset" || exit 1
wget 'https://huggingface.co/api/datasets/TrainingDataPro/brain-mri-dataset/parquet/default/train/0.parquet'

cd "$REG/body-measurements-dataset" || exit 1
wget "https://huggingface.co/api/datasets/TrainingDataPro/body-measurements-dataset/parquet/default/train/0.parquet"