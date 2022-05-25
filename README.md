Non-Autoregressive Unsupervised Summarizer (NAUS)
=======
This repo contains the code for our ACL 2022 paper [Learning Non-Autoregressive Models from Search for Unsupervised Sentence Summarization](https://aclanthology.org/2022.acl-long.545).

## Additional Results: Supervised Summarization

<div align="center">
	Table1. Model performance on the Gigaword headline generation test dataset in the supervised setting
	
<table>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;Models</td>
		<td>Parameters</td>
		<td>Row #</td>
		<td>Rouge-1</td>
		<td>Rouge-2</td>
		<td>Rouge-L</td>
		<td>Avg Rouge</td>
		<td>Len</td>
	</tr>
	<tr>
		<td rowspan="8"><a href=https://github.com/yxuansu/NAG-BERT>&nbsp;NAG-BERT</a></td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.2</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;1</td>
		<td>&nbsp;&nbsp;29.05</td>
		<td>&nbsp;&nbsp;12.69</td>
		<td>&nbsp;&nbsp;27.52</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;23.09</td>
		<td>6.2</td>
	</tr>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.3</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;2</td>
		<td>&nbsp;&nbsp;30.05</td>
		<td>&nbsp;&nbsp;13.80</td>
		<td>&nbsp;&nbsp;28.87</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;24.24</td>
		<td>6.6</td>
	</tr>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.4</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;3</td>
		<td>&nbsp;&nbsp;30.47</td>
		<td>&nbsp;&nbsp;13.58</td>
		<td>&nbsp;&nbsp;28.81</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;24.29</td>
		<td>6.7</td>
	</tr>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.5</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;4</td>
		<td>&nbsp;&nbsp;30.41</td>
		<td>&nbsp;&nbsp;13.53</td>
		<td>&nbsp;&nbsp;28.63</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;24.19</td>
		<td>6.7</td>
	</tr>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.6</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;5</td>
		<td>&nbsp;&nbsp;30.61</td>
		<td>&nbsp;&nbsp;13.55</td>
		<td>&nbsp;&nbsp;28.97</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;24.38</td>
		<td>6.8</td>
	</tr>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.7</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;6</td>
		<td>&nbsp;&nbsp;30.30</td>
		<td>&nbsp;&nbsp;13.59</td>
		<td>&nbsp;&nbsp;28.67</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;24.19</td>
		<td>6.8</td>
	</tr>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.8</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;7</td>
		<td>&nbsp;&nbsp;30.21</td>
		<td>&nbsp;&nbsp;13.05</td>
		<td>&nbsp;&nbsp;28.59</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;23.95</td>
		<td>6.8</td>
	</tr>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.9</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;8</td>
		<td>&nbsp;&nbsp;30.57</td>
		<td>&nbsp;&nbsp;13.64</td>
		<td>&nbsp;&nbsp;28.99</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;24.40</td>
		<td>6.8</td>
	</tr>
	<tr>
		<td rowspan="2">&nbsp;NAUS+LC</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.23</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;9</td>
		<td>&nbsp;&nbsp;33.73</td>
		<td>&nbsp;&nbsp;13.26</td>
		<td>&nbsp;&nbsp;31.68</td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;26.22</td>
		<td>6.4</td>
	</tr>
	<tr>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.24</td>
		<td>&nbsp;&nbsp;&nbsp;10</td>
		<td>&nbsp;&nbsp;<b>34.56</b></td>
		<td>&nbsp;&nbsp;<b>14.10</b></td>
		<td>&nbsp;&nbsp;<b>32.45</b></td>
		<td>&nbsp;&nbsp;&nbsp;&nbsp;<b>27.04</b></td>
		<td>6.8</td>
	</tr>

</table>

</div>


## Environment setup
The scripts are developed with [Anaconda](https://www.anaconda.com/) python 3.8, and the working environment can be configured with the following commands. 

```
git clone https://github.com/MANGA-UOFA/NAUS
cd NAUS
conda create -n NAUS_MANGA python=3.8

conda activate NAUS_MANGA

pip install gdown
pip install git+https://github.com/tagucci/pythonrouge.git
conda install pytorch cudatoolkit=10.2 -c pytorch

git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ..
rm -rf ctcdecode

pip install -e.
```

## Data downloading
The [search-based](https://aclanthology.org/2020.acl-main.452.pdf) summaries on the Gigaword dataset and pre-trained model weights can be found in this publically available [Google drive folder](https://drive.google.com/drive/folders/1XKN6oFy2-C6ChkfjUVIJHXFCqTVF9vjo), which can be automatically downloaded and organized with the following commands. 

```
chmod +x download_data.sh
./download_data.sh
```

## Preprocess
Execute the following command to preprocess the data.

```
chmod +x preprocess_giga.sh
./preprocess_giga.sh
```


## Model training
Our training script is ```train.py```. We introduce some of its important training parameters, other parameters can be found [here](https://fairseq.readthedocs.io/en/latest/command_line_tools.html).

**data_source**: (Required) Directory to the pre-processed training data (e.g., data-bin/gigaword_10).

**arch**: (Required) Model Architecture. This must be set to ```nat_encoder_only_ctc```.

**task**: (Required) The task we are training for. This must be set to ```summarization```.

**best-checkpoint-metric**: (Required) Criteria to save the best checkpoint. This can be set to either ```rouge``` or ```loss``` (we used rouge).

**criterion**: (Required) Criteria for training loss calculation. This must be set to ```summarization_ctc```. 

**max-valid-steps**: (Optional) Maximum steps during validation. e.g., ```100```. Limiting this number avoids time-consuming validation on a large validation dataset. 

**batch-size-valid** (Optional) Batch size during validation. e.g., ```5```. Set this parameter to ```1``` if you want to test the **unparallel** inference efficiency. 

**decoding_algorithm**: (Optional) Decoding algorithm of model output (logits) sequence. This can be set to ```ctc_greedy_decoding```, ```ctc_beam_search``` and ```ctc_beam_search_length_control```.

**truncate_summary**: (Optional) Whether to truncate the generated summaries. This parameter is valid when ```decoding_algorithm``` is set to ```ctc_greedy_decoding``` or ```ctc_beam_search```.

**desired_length**: (Optional) Desired (maximum) length of the output summary. If ```decoding_algorithm``` is set to ```ctc_greedy_decoding``` or ```ctc_beam_search``` and ```truncate_summary``` is ```True```, the model will truncate longer summaries to the ```desired_length```.
When ```decoding_algorithm``` is  ```ctc_beam_search_length_control```, the model's decoding strategy depends on the parameter ```force_length```, which will be explained in the next paragraph. 

**force_length**: (Optional) This parameter is only useful when ```decoding_algorithm``` is set to ```ctc_beam_search_length_control```.
For ```ctc_beam_search_length_control```, the parameter determines whether to force the length of the generated summaries to be ```desired_length```. If ```force_length``` is set to ```False```, the model returns the greedily decoded summary if the summary length does not exceed ```desired_length```. Otherwise, the model search for the (approximately) most probable summary of the ```desired_length``` with a [length control](https://openreview.net/forum?id=UNzc8gReN7m) algorithm. 

**beam_size**: (Optional) Beam size for the decoding algorithm, only useful when ```decoding_algorithm``` is set to ```ctc_beam_search``` or ```ctc_beam_search_length_control```.

**valid_subset**: (Optional) Names of the validation dataset, separating by comma, e.g, test,valid.

**max_token**: (Optional) Max tokens in each training batch.

**max_update**: (Optional) Maximum training steps.


For example, if we want to train NAUS with 10-word HC summaries, ```ctc_beam_search_length_control``` decoding, desired length of ```10``` and beam size of ```6```, we can use the following training command. 

```
data_source=gigaword_10
decoding_algorithm=ctc_beam_search_length_control
desired_length=10
beam_size=6
valid_subset=valid
drop_out=0.1
max_token=4096
max_update=50000
use_length_ratio=False
force_length=False
truncate_summary=False
max_valid_steps=100
label_smoothing=0.1

CUDA_VISIBLE_DEVICES=0 nohup python train.py data-bin/$data_source --source-lang article --target-lang summary --save-dir NAUS_${data_source}_${max_token}_${decoding_algorithm}_${desired_length}_beam_size_${beam_size}_truncate_summary_${truncate_summary}_use_length_ratio_${use_length_ratio}_label_smoothing_${label_smoothing}_dropout_${drop_out}_checkpoints --keep-interval-updates 5 --save-interval-updates 5000 --validate-interval-updates 5000 --scoring rouge --maximize-best-checkpoint-metric --best-checkpoint-metric rouge --log-format simple --log-interval 100 --keep-last-epochs 5 --keep-best-checkpoints 5 --share-all-embeddings --encoder-learned-pos --optimizer adam --adam-betas "(0.9,0.98)" --lr 0.0005 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 10000 --warmup-init-lr 1e-07 --weight-decay 0.01 --fp16 --clip-norm 2.0 --max-update $max_update --task summarization --criterion summarization_ctc --arch nat_encoder_only_ctc --activation-fn gelu --dropout 0.1 --max-tokens $max_token --valid-subset $valid_subset --decoding_algorithm $decoding_algorithm --desired_length $desired_length --beam_size $beam_size --use_length_ratio $use_length_ratio --force_length $force_length --truncate_summary $truncate_summary --max-valid-steps $max_valid_steps&
```

## Model evaluation
Our evaluation script is ```fairseq_cli/generate.py```, and it inherits the training arguments related to the data source, model architecture and decoding strategy.
Besides, it requires the following arguments. 

**path**: (Required) Directory to the trained model (e.g., NAUS/checkpoint_best.pt).

**gen-subset**: (Required) Names of the generation dataset (e.g., test). 

**scoring**: (Required) Similar to the criteria in training arguments, it must be set to ```rouge```.


For example, the following command evaluates the performance of our pretrained model```HC_10_length_control.pt``` on the Gigaword test dataset.

```
data_source=data-bin/gigaword_10
path=model_weights/HC_10_length_control.pt
seed=17
gen_subset=test

CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py $data_source --seed $seed --source-lang article --target-lang summary --path $path  --task summarization --scoring rouge --arch nat_encoder_only_ctc --gen-subset $gen_subset
```

The evaluation result will be saved at the folder ```*_evaluation_result``` by default, including the generated summaries and the statistics of the generated summaries (e.g., ROUGE score).

Notice: if you want to test the **unparallel** inference efficiency, include an extra parameter ```--batch-size 1``` in the evaluation command.

## Final comments
As you may notice, our script is developed based on [Fairseq](https://github.com/pytorch/fairseq), which is a very useful & extendable package to develop Seq2Seq models. We didn't ruin any of its built-in functionality to retain its extension ability. 
