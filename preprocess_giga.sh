#!/bin/bash
python create_giga_HC_dictionary.py

TEXT=gigaword_8
python fairseq_cli/preprocess.py --source-lang article --target-lang summary --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test,$TEXT/DUC2004 --destdir data-bin/$TEXT --workers 40 --joined-dictionary --task summarization --srcdict giga_HC.dict

mv data-bin/$TEXT/test1.article-summary.article.bin data-bin/$TEXT/duc2004.article-summary.article.bin
mv data-bin/$TEXT/test1.article-summary.article.idx data-bin/$TEXT/duc2004.article-summary.article.idx
mv data-bin/$TEXT/test1.article-summary.summary.bin data-bin/$TEXT/duc2004.article-summary.summary.bin
mv data-bin/$TEXT/test1.article-summary.summary.idx data-bin/$TEXT/duc2004.article-summary.summary.idx

TEXT=gigaword_10
python fairseq_cli/preprocess.py --source-lang article --target-lang summary --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test,$TEXT/DUC2004 --destdir data-bin/$TEXT --workers 40 --joined-dictionary --task summarization --srcdict giga_HC.dict

mv data-bin/$TEXT/test1.article-summary.article.bin data-bin/$TEXT/duc2004.article-summary.article.bin
mv data-bin/$TEXT/test1.article-summary.article.idx data-bin/$TEXT/duc2004.article-summary.article.idx
mv data-bin/$TEXT/test1.article-summary.summary.bin data-bin/$TEXT/duc2004.article-summary.summary.bin
mv data-bin/$TEXT/test1.article-summary.summary.idx data-bin/$TEXT/duc2004.article-summary.summary.idx

TEXT=gigaword_13
python fairseq_cli/preprocess.py --source-lang article --target-lang summary --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test,$TEXT/DUC2004 --destdir data-bin/$TEXT --workers 40 --joined-dictionary --task summarization --srcdict giga_HC.dict

mv data-bin/$TEXT/test1.article-summary.article.bin data-bin/$TEXT/duc2004.article-summary.article.bin
mv data-bin/$TEXT/test1.article-summary.article.idx data-bin/$TEXT/duc2004.article-summary.article.idx
mv data-bin/$TEXT/test1.article-summary.summary.bin data-bin/$TEXT/duc2004.article-summary.summary.bin
mv data-bin/$TEXT/test1.article-summary.summary.idx data-bin/$TEXT/duc2004.article-summary.summary.idx

python create_giga_ref_dictionary.py
TEXT=gigaword_ref
python fairseq_cli/preprocess.py --source-lang article --target-lang summary --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test,$TEXT/DUC2004 --destdir data-bin/$TEXT --workers 40 --joined-dictionary --task summarization --srcdict giga_ref.dict

mv data-bin/$TEXT/test1.article-summary.article.bin data-bin/$TEXT/duc2004.article-summary.article.bin
mv data-bin/$TEXT/test1.article-summary.article.idx data-bin/$TEXT/duc2004.article-summary.article.idx
mv data-bin/$TEXT/test1.article-summary.summary.bin data-bin/$TEXT/duc2004.article-summary.summary.bin
mv data-bin/$TEXT/test1.article-summary.summary.idx data-bin/$TEXT/duc2004.article-summary.summary.idx
