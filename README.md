# SWEET
Separating Weights for Early-Exit Transformers

This repository contains the code used for experiments in the paper  
 
 [**Finding the SWEET spot:** Improvement and Anlysis of Adaptive Inference in Low Resource Settings''](insert link upon publication)
 
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
We provide 
### Early Exit - SWEET

### Multi Model
