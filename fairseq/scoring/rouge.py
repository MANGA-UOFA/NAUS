# Copyright (c) Puyuan Liu
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pythonrouge.pythonrouge import Pythonrouge
import ctypes
from dataclasses import dataclass, field

from argparse import Namespace
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import register_scorer
import numpy as np


class ROUGEStat(ctypes.Structure):
    _fields_ = [
        ("ROUGE-1-R", ctypes.c_size_t),
        ("ROUGE-1-R-cf95", ctypes.c_size_t),
        ("ROUGE-1-P", ctypes.c_size_t),
        ("ROUGE-1-F", ctypes.c_size_t),
        ("ROUGE-1-F-cf95", ctypes.c_size_t),

        ("ROUGE-2-R", ctypes.c_size_t),
        ("ROUGE-2-R-cf95", ctypes.c_size_t),
        ("ROUGE-2-P", ctypes.c_size_t),
        ("ROUGE-2-F", ctypes.c_size_t),
        ("ROUGE-2-F-cf95", ctypes.c_size_t),

        ("ROUGE-L-R", ctypes.c_size_t),
        ("ROUGE-L-R-cf95", ctypes.c_size_t),
        ("ROUGE-L-P", ctypes.c_size_t),
        ("ROUGE-L-F", ctypes.c_size_t),
        ("ROUGE-L-F-cf95", ctypes.c_size_t),

        ("Source-Average-Length", ctypes.c_size_t),
        ("HypSummary-Average-Length", ctypes.c_size_t),
        ("RefSummary-Average-Length", ctypes.c_size_t),
    ]


@dataclass
class ROUGEConfig(FairseqDataclass):
    ROUGE_n_gram: int = field(default=2, metadata={"help": "Compute ROUGE-N up to max-ngram length will be computed."})
    ROUGE_SU4: bool = field(default=False, metadata={"help": "Compute ROUGE-SU4 measures unigram and skip-bigram."})
    ROUGE_stemming: bool = field(default=True, metadata={"help": "Stem both model and system summaries using "
                                                                 "Porter stemmer before computing various statistics."})
    ROUGE_stopwords: bool = field(default=False, metadata={"help": "Remove stopwords in model and system summaries "
                                                                   "before computing various statistics."})
    ROUGE_word_level: bool = field(default=True, metadata={"help": "Evaluate based on words. If False, rouge "
                                                                   "evaluates the system summary based on bytes."})
    ROUGE_length_limit: bool = field(default=False, metadata={"help": "If you want to limit the length of the system "
                                                                      "summary,set True."})
    ROUGE_length: int = field(default=100, metadata={"help": "Limit first N words/bytes of the system summary."})

    ROUGE_use_cf: bool = field(default=True, metadata={"help": "If True, you can use confidence interval to compute."})
    ROUGE_cf: int = field(default=95, metadata={"help": "Confidence interval (default is 95%)."})
    ROUGE_scoring_formula: str = field(default="average", metadata={"help": "'average' is calculated by model average. "
                                                                            "'best' is calculated by best model."})
    ROUGE_resampling: bool = field(default=True, metadata={"help": "Use bootstrap resampling."})
    ROUGE_samples: int = field(default=1000, metadata={"help": "specify the number of sampling point in bootstrap "
                                                               "resampling(default is 1000)."})
    ROUGE_favor: bool = field(default=True, metadata={"help": "If True, set relative importance of ROUGE scores "
                                                              "as blow."})
    ROUGE_p: float = field(default=0.5, metadata={"help": "Relative importance of recall and precision ROUGE scores."})

    ROUGE_mean: str = field(default="arithmetic", metadata={"help": "Method to calculate the average ROUGE score. "
                                                                    "Can be chosen from geometric and arithmetic"})


@register_scorer("rouge", dataclass=ROUGEConfig)
class Scorer(object):
    def __init__(self, cfg: Namespace) -> None:
        self.stat = ROUGEStat()
        self.n_gram = cfg.ROUGE_n_gram
        self.ROUGE_SU4 = cfg.ROUGE_SU4
        self.stemming = cfg.ROUGE_stemming
        self.stopwords = cfg.ROUGE_stopwords
        self.word_level = cfg.ROUGE_word_level
        self.length_limit = cfg.ROUGE_length_limit
        self.length = cfg.ROUGE_length
        self.use_cf = cfg.ROUGE_use_cf
        self.cf = cfg.ROUGE_cf
        self.scoring_formula = cfg.ROUGE_scoring_formula
        self.resampling = cfg.ROUGE_resampling
        self.samples = cfg.ROUGE_samples
        self.favor = cfg.ROUGE_favor
        self.p = cfg.ROUGE_p
        self.mean = cfg.ROUGE_mean

    def calculate_score(self, source_text, hypothesis, ref_summary):
        rouge = Pythonrouge(summary_file_exist=False, summary=[[hyp] for hyp in hypothesis],
                            reference=[[[ref]] for ref in ref_summary], n_gram=self.n_gram, ROUGE_SU4=self.ROUGE_SU4,
                            ROUGE_L=True, stemming=self.stemming, stopwords=self.stopwords, word_level=self.word_level,
                            length_limit=self.length_limit, length=self.length, use_cf=self.use_cf, cf=self.cf,
                            scoring_formula=self.scoring_formula, resampling=self.resampling, samples=self.samples,
                            favor=self.favor, p=self.p)

        print("Calculating ROUGE score...\n")
        rouge_scores_dict = rouge.calc_score()
        rouge_f_list = [rouge_scores_dict["ROUGE-1-F"], rouge_scores_dict["ROUGE-2-F"], rouge_scores_dict["ROUGE-L-F"]]
        rouge_arithmetic_mean = np.mean(rouge_f_list)
        rouge_geometric_mean = np.exp(np.log(rouge_f_list).mean())
        if self.mean == "arithmetic":
            mean_rouge = rouge_arithmetic_mean
        elif self.mean == "geometric":
            mean_rouge = rouge_geometric_mean
        else:
            raise (NotImplementedError, "Only arithmetic and geometric means are supported as the rouge validation"
                                        "criteria for now")
        rouge_scores_dict["arithmetic_rouge"] = rouge_arithmetic_mean
        rouge_scores_dict["geometric_rouge"] = rouge_geometric_mean
        rouge_scores_dict["rouge_mean"] = mean_rouge  # This is for validation criteria.
        print("Successfully calculated ROUGE score. \n")

        # print("Calculating average summary length...\n")
        source_text_length_list = []
        hyp_summary_length_list = []
        hyp_summary_char_length_list = []
        ref_summary_length_list = []
        for i in range(0, len(hypothesis)):
            source_text_length_list.append(len(source_text[i].split()))
            hyp_summary_length_list.append(len(hypothesis[i].split()))
            ref_summary_length_list.append(len(ref_summary[i].split()))
            hyp_summary_char_length_list.append(len(hypothesis[i]))
        length_dict = {"source_ave_length": sum(source_text_length_list) / len(source_text_length_list),
                       "hyp_ave_length": sum(hyp_summary_length_list) / len(hyp_summary_length_list),
                       "ref_ave_length": sum(ref_summary_length_list) / len(ref_summary_length_list),
                       "hyp_ave_char_length": sum(hyp_summary_char_length_list) / len(hyp_summary_char_length_list)}

        return rouge_scores_dict, length_dict
