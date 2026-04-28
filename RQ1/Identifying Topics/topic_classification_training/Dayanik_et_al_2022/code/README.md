
## Dependencies & Installation
- Note that the code for ILP decoding relies on a python library called 'PySCIPOpt' which requires a working installation of the SCIP Optimization Suite. Please, make sure that your SCIP installation works before running the command below.

 ``` 
 pip install -r requirements.txt 
 ```

## Dataset
- We are not allowed to redistribute the data. It can be obtained through the MARDY web site, https://mardy-spp.github.io/

- Please make sure that it is placed under `data` folder:


## To train the models:
```
# Base model:
CUDA_VISIBLE_DEVICES=0 python bert_main_wo_reg.py plain bert_plain_test_output.csv 0.3 5e-5
# HLE Model:
CUDA_VISIBLE_DEVICES=0 python bert_main_wo_reg.py hle bert_hle_test_output.csv 0.3 5e-5
# CRR Model:
CUDA_VISIBLE_DEVICES=0 python bert_main.py plain bert_crr_test_output.csv 0.005 -0.01 0.4 5e-5
# HLE+CRR Model:
CUDA_VISIBLE_DEVICES=0 python bert_main.py hle bert_hle_crr_test_output.csv 0.01 -0.01 0.4 5e-5

# ILP Model:
python ILP_decoder.py bert_plain_test_output.csv 
# HLE+ILP Model:
python ILP_decoder.py bert_hle_test_output.csv   
# CRR+ILP Model:
python ILP_decoder.py bert_crr_test_output.csv   
# HLE+CRR+ILP Model:
python ILP_decoder.py bert_hle_crr_test_output.csv   
```
   
