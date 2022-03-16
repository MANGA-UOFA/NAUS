# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# This script intends to manually create a dictionary for the training data

from tqdm import tqdm
from collections import Counter
# Since the training target are fully extractive, we only consider words in the training source
train_source_text = list(open("gigaword_8/train.article", "r"))
# We also consider the test target as we do not want to replace any words in it by unk token.
test_target_text = list(open("gigaword_8/test.summary"))
# Combine two text file together
total_text = train_source_text + test_target_text
# Create an empty dictionary
d = dict()

# Loop through each line of the file
for line in tqdm(total_text):
    # Remove the leading spaces and newline character
    line = line.strip()

    # Convert the characters in line to
    # lowercase to avoid case mismatch
    line = line.lower()

    # Split the line into words
    words = line.split(" ")

    # Iterate over each word in line
    for word in words:
        # Check if the word is already in dictionary
        if word in d:
            # Increment count of word by 1
            d[word] = d[word] + 1
        else:
            # Add the word to dictionary with count 1
            d[word] = 1

# Save the words and occurrence to a dictionary
dict_name = "giga_HC.dict"
d = Counter(d)
with open(dict_name, "w+") as f:
    for key, value in list(d.most_common()):
        if key == "<|unk|>":
            # We don't include <|unk|> to the dict since our ctc dictionary will later add this
            continue
        f.write("%s %d\n" % (key, value))

print("Successfully created dictionary %s" % dict_name)