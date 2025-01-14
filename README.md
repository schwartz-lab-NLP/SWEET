# SWEET
Separating Weights for Early-Exit Transformers

This repository contains the code used for experiments in the paper  
 
 [**Finding the SWEET spot:** Improvement and Anlysis of Adaptive Inference in Low Resource Settings'']([arxiv](https://arxiv.org/abs/2306.02307))
 
<p align="center">
    <img src=/figures/SWEET_illustration.png  width=40% height=40% align="center" alt="Illustration of the SWEET method">
</p>
 
 
OUr code is implemented using the [HuggingFace](https://huggingface.co/) framework.
## Setup 

To run the code, follow these instructions: 
First, clone Huggingface's transformer library ([found here](https://github.com/huggingface/transformers)) and install all requirements. Instructions can be found [here](https://huggingface.co/docs/transformers/installation#editable-install)

Once everything is set up, clone this repository and follow these 4 simple steps: 
1) copy the two scripts found in /SWEET/code/scripts into /transformers/examples/pytorch/text-classification.
2) copy three util files from /SWEET/code/utils into /transformers/src/transformers/utils.
3) Copy the two altered bert files from /SWEET/code/models/bert into /transformers/src/transformers/models/bert.
4) Copy the two altered deberta files from /SWEET/code/models/deberta into /transformers/src/transformers/models/deberta. 


## Execution
We provide instructions for training EE amd MM bert models 
### Early Exit - SWEET
Navigate to the directory contatining the _run_glue_ files:
```bash 
cd /transformers/examples/pytorch/text-classification
```
one in the correct filder, run: 
```bash 
python3 run_glue_EE.py --model_name_or_path bert-base-uncased --task_name mnli --per_device_train_batch_size 16 --per_device_eval_batch_size 1 --do_train --do_calibration --do_eval --max_seq_length 256 --max_train_samples 6000 --output_dir ${OUTPUT_DIR} --cache_dir ${CACHE_DIR} --learning_rate 5e-5 --exit_layers 0_3_5_11  --exit_threshold 11 --num_train_epochs 2 --SWEET
```

* To run standard early exiting (without SWEET), disable the SWEET flag at the end of the line

### Multi Model

```bash 
python3 run_glue_MM.py --model_name_or_path bert-base-uncased --task_name mnli --per_device_train_batch_size 16 --per_device_eval_batch_size 1 --do_train --do_calibration --do_eval --max_seq_length 256 --max_train_samples 6000 --output_dir ${OUTPUT_DIR} --cache_dir ${CACHE_DIR} --learning_rate 5e-5 --exit_layers 0_3_5_11  --exit_threshold 11 --num_train_epochs 2
```
* Insert your own output_dir and cache_dir arguments to make the doce compatible with your local machine
* For running deberta, use --model_name_or_path microsoft/deberta-base




