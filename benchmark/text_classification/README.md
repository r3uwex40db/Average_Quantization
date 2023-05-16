# GACT_AQ NLP Experiments
## Requirements
Make sure you have GACT_AQ installed. 
Install other dependency with the following command:
```bash
pip install -r requirements.txt
```

## Finetune models 
### Benchmark accuracy
```bash
python run_glue.py --model_name_or_path ARCH
--task_name TASK --max_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 128 --learning_rate 1e-5 --num_train_epochs 1 --seed 42 --pad_to_max_length  --output_dir log/TASK/LEVEL/ --gact --opt_level LEVEL --bit BIT --aq-bit AQ-BIT --device DEVICE_NUM
```
The choices for TASK are {cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli}. In the papaer, we experiment with mnli, sst2,  and qnli datasets.

The choices for ARCH are defined in huggingface.co/models. In the paper, we experiment with ```bert-large-cased```.

The choices for LEVEL are {L1, L1.1, L1.2, L2, L2.1, L2.2, L3}. In the paper, we epxperiment with ```L2.2```.

In the paper, for each seed, we experiment with different learning rates (0.5e-5 1e-5 1.5e-5 2e-5), we pick the highest accuracy among different learning rates. The final accuracy is the average among three seeds (42, 43, 44). 


### Benchmark memory
Add `--get-mem` to the end of the command. For example, to get the training memory of GACT_AQ-0.5bit when target precision is set to 1-bit on bert-large-cased sst2 dataset, run the following command:
```bash
python run_glue.py --model_name_or_path bert-base-cased --task_name sst2 --max_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 128 --learning_rate 1e-5 --num_train_epochs 1 --seed 42 --pad_to_max_length  --output_dir log/sst2/L2/ --gact --opt_level L2 --bit 2 --aq-bit 0.5 --get-mem
```

### Benchmark speed
The benchmark will run the first five iterations and calculate the trainning speed of each iteration. The reported speed is the median of the five speed data points.

To get the training speed, add `--get-speed` to the end of the command. For example, to get the training speed of GACT_AQ-0.5bit when target precision is set to 1-bit on bert-large-cased sst2 dataset, run the following command:
```bash
python run_glue.py --model_name_or_path bert-base-cased --task_name sst2 --max_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 128 --learning_rate 1e-5 --num_train_epochs 1 --seed 42 --pad_to_max_length  --output_dir log/sst2/L2/ --gact --opt_level L2 --bit 2 --aq-bit 0.5 --get-speed
```
Notice GACT_AQ will calculate the layer sensitivity on the first iteration, therefore the first iteration will be slower than the following iterations.

### Find the biggest model with full precision/GACT
```bash
python exp_mem_speed.py --mode binary_search_max_hidden_size
python exp_mem_speed.py --mode binary_search_max_layer
python exp_mem_speed.py --mode binary_search_max_intermediate_size
```
The results will be in `max_hidden_size_results.json`, `max_layer_results.json`, `max_intermediate_results.json` respectively.


