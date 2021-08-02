from src.utils import get_summary_paths, read_label
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit, Punctuation, Sequence
from tokenizers.normalizers import Lowercase, NFKC, Replace, Strip
from tokenizers import normalizers
from pathlib import Path
from string import punctuation, digits
from nltk.corpus import stopwords


# prepare filepaths
CWD = Path.cwd()
DATA = Path('/work/fabiasch/eia/data')
SEG = DATA.joinpath("filtered2")

# prepare tokenizer
replace_punctuation = [Replace(x, "") for x in punctuation]
replace_bullet = Replace("·", "")
replace_digits = [Replace(x, "") for x in digits]
normalizer = normalizers.Sequence(
    [NFKC(), Strip(), Lowercase(), *replace_digits, *replace_punctuation, replace_bullet]
)
pre_tokenizer = Sequence(
    [
        WhitespaceSplit(),
    ]
)
tokenizer = Tokenizer(WordLevel())
tokenizer.add_special_tokens(['<unk>'])
trainer = WordLevelTrainer(
    vocab_size=20_000,
    special_tokens=["<unk>"],
)


label, _ = read_label("./label_extra_pos.csv")
files = [str(x) for x in get_summary_paths(SEG)]
files = [file for file in files if any(cod in str(file) for cod in label)]

tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.normalizer = normalizer
tokenizer.train(files, trainer)
tokenizer.save(f"tokenizer.lr.json", pretty=True)
