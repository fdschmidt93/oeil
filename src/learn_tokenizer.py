from pathlib import Path
from string import digits, punctuation

from nltk.corpus import stopwords
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import NFKC, Lowercase, Replace, Strip, StripAccents
from tokenizers.pre_tokenizers import Punctuation, Sequence, WhitespaceSplit
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer

from src.utils import get_summary_paths, read_label


def learn_tokenize(files, path=f"tokenizer.lr.json"):
    # prepare tokenizer
    replace_punctuation = [Replace(x, "") for x in punctuation]
    replace_bullet = Replace("·", "")
    replace_digits = [Replace(x, "") for x in digits]
    normalizer = normalizers.Sequence(
        [
            NFKC(),
            Strip(),
            # Lowercase(),
            StripAccents(),
            *replace_digits,
            *replace_punctuation,
            replace_bullet,
        ]
    )
    pre_tokenizer = Sequence(
        [
            WhitespaceSplit(),
        ]
    )
    tokenizer = Tokenizer(WordLevel())
    tokenizer.add_special_tokens(["<unk>"])
    trainer = WordLevelTrainer(
        vocab_size=20_000,
        special_tokens=["<unk>"],
    )

    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.normalizer = normalizer
    tokenizer.train([str(x) for x in files], trainer)
    tokenizer.save(str(path), pretty=True)
    return tokenizer

# if __name__ == "__main__":
    # # prepare filepaths
    # CWD = Path.cwd()
    # DATA = Path("./data/summaries/preprocessed")

    # label, _ = read_label("./data/labels.csv")
    # files = [str(x) for x in get_summary_paths(DATA, final_act=False)]
    # files = [file for file in files if any(cod in str(file) for cod in label)]
    # assert len(label) == len(files)
    # learn_tokenize(files)
