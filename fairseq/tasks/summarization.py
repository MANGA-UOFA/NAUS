# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import logging
import os
from typing import Optional
from omegaconf import II
import torch
from fairseq.utils import new_arange
import numpy as np
from fairseq import metrics, utils, tokenizer
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    SummarizationDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import CTCDictionary
from fairseq.scoring.rouge import Scorer as RougeScorer
import time

logger = logging.getLogger(__name__)
NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])
GENERATOR_CHOICES = ChoiceEnum(["ctc", "at", "nat"])


def load_summarization_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    keep_bos_eos_tgt=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return SummarizationDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        keep_bos_eos_tgt=keep_bos_eos_tgt,
    )


@dataclass
class SummarizationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=False, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")
    use_bpe: bool = field(
        default=False, metadata={"help": "whether bpe is used to tokenize the train/valid/test data"}
    )

    # options for reporting rouge during validation
    # eval_rouge: bool = field(
    #     default=True, metadata={"help": "evaluation with ROUGE scores"}
    # )

    noise: NOISE_CHOICES = field(
        default="random_delete",
        metadata={"help": "type of noise"},
    )

    upsample_primary: int = field(default=1, metadata={"help": "upsampling the source text"})

    generator_type: GENERATOR_CHOICES = field(
        default="ctc",
        metadata={"help": "type of generator"},
    )

    keep_bos_eos_tgt: bool = field(
        default=False, metadata={"help": "whether keep the bos and eos of targets"}
    )


@register_task("summarization", dataclass=SummarizationConfig)
class SummarizationTask(FairseqTask):
    """
    Generate summary (target) for am input sentence (source).

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target
        Currently, we set the dictionary of target and source to be the same for this summarization task

    .. note::
        This script is created based on the translation script. Since both translation and summarization is about text
        generation, their workflow is similar.

        The summarization task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: SummarizationConfig

    def __init__(self, cfg: SummarizationConfig, src_dict, tgt_dict, use_ctc=True):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = src_dict  # Force the source and target dictionary to be the same
        self.use_ctc = use_ctc

        # Store the generated summaries and stats during the validation
        self.source_text_list = []
        self.hyp_summary_list = []
        self.ref_summary_list = []
        self.inference_time_list = []
        self.RougeScorer = None

    @classmethod
    def load_dictionary(cls, filename):
        """Overwrite the load_dictionary function of fairseq to incorporate with CTC nicely.
        Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return CTCDictionary.load(filename)

    @classmethod
    def setup_task(cls, cfg: SummarizationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        # we only need src_dict, we keep both tgt and src to avoid some adaption issue.
        return cls(cfg, src_dict, tgt_dict)

    def decode_to_vocabulary(self, sentence_index):
        """
        Decode 2D list to token index into actual tokens
        """
        decoded_result = [self.tgt_dict.string(i, unk_string=("<|unk|>"), extra_symbols_to_ignore=[self.tgt_dict.pad()]).
                              strip("\n") for i in sentence_index]
        return decoded_result

    def setup_rouge(self, cfg):
        self.RougeScorer = RougeScorer(cfg)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = CTCDictionary()
        for filename in filenames:
            CTCDictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_summarization_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
            keep_bos_eos_tgt=self.cfg.keep_bos_eos_tgt
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the summarization task is not supported"
            )

        return SummarizationDataset(src_tokens, src_lengths, self.source_dictionary, append_bos=True)

    def build_model(self, cfg):
        model = super().build_model(cfg)
        self.generator = self.build_generator([model], None)  # We disable generator config for now
        return model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        sample["prev_target"] = self.inject_noise(sample["target"])
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        sample_size = 1  # Avoid mis-counting of the fairseq trainer
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            # sample["prev_target"] = self.inject_noise(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample)
            source_texts = sample["net_input"]["src_tokens"]
            ref_summaries = sample["target"]
            # Notice target dictionary is the same as the source dictionary in summarization task
            decoded_source_texts = self.decode_to_vocabulary(source_texts.tolist())
            formatted_hyp_summaries = self.generator.generate([model], sample)
            hyp_summaries = [sentence[0]["tokens"] for sentence in formatted_hyp_summaries]
            inference_time = formatted_hyp_summaries[0][0]["inference_time"]  # Each sample stored the total infer time
            decoded_hyp_summaries = self.decode_to_vocabulary(hyp_summaries)
            decoded_ref_summaries = self.decode_to_vocabulary(ref_summaries.tolist())
            self.record_summary(decoded_source_texts, decoded_hyp_summaries, decoded_ref_summaries, inference_time)

        return loss, sample_size, logging_output

    def record_summary(self, source_text, hyp_summary, ref_summary, inference_time):
        """
        Record the generated summaries during the validation such that those summaries can be latter saved
        """
        self.source_text_list.append(source_text)
        self.hyp_summary_list.append(hyp_summary)
        self.ref_summary_list.append(ref_summary)
        self.inference_time_list.append(inference_time)

    def clear_recorded_summary(self):
        """
        Clear the recorded summaries.
        """
        self.source_text_list = []
        self.hyp_summary_list = []
        self.ref_summary_list = []
        self.inference_time_list = []

    def reduce_recorded_summary_list(self):
        """
        concatenate the generated list of summaries.
        """
        self.source_text_list = np.concatenate(self.source_text_list).tolist()
        self.hyp_summary_list = np.concatenate(self.hyp_summary_list).tolist()
        self.ref_summary_list = np.concatenate(self.ref_summary_list).tolist()

    def save_summary(self, save_dir, subset, epoch, n_iter, generate=False):
        """
        Save the recorded summaries into files.
        """
        if generate == False:
            save_path = "epoch_%d_step_%d" % (epoch, n_iter)
        else:
            save_path = ""
        save_path = os.path.join(save_dir, save_path)
        save_path = os.path.join(save_path, subset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, "article.txt"), "w+") as f:
            for item in self.source_text_list:
                f.write("%s\n" % item)
        with open(os.path.join(save_path, "generated_summary.txt"), "w+") as f:
            for item in self.hyp_summary_list:
                f.write("%s\n" % item)
        with open(os.path.join(save_path, "ref_summary.txt"), "w+") as f:
            for item in self.ref_summary_list:
                f.write("%s\n" % item)

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.summarization_ctc_generator import SummarizationCTCGenerator
        from fairseq.summarization_at_generator import SummarizationATGenerator

        if models[0].cfg.generator_type == "ctc":
            generator = SummarizationCTCGenerator(self.target_dictionary, models, retain_dropout=False, adaptive=False)
        elif models[0].cfg.generator_type == "at":
            generator = SummarizationATGenerator(models, self.target_dictionary,
                                                 max_len_b=models[0].cfg.desired_length,
                                                 beam_size=models[0].cfg.beam_size)
        else:
            raise NotImplementedError("Cannot initialize generator of type %s" % self.cfg.generator_type)

        return generator

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def get_statistics(self):
        """
        Get some statistics of the generated summary, including the summary length, ROUGE score and inference time
        """
        rouge_stat_dict, length_dict = self.RougeScorer.calculate_score(self.source_text_list, self.hyp_summary_list,
                                                             self.ref_summary_list)

        return rouge_stat_dict, length_dict
