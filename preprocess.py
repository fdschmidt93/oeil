import re
from pathlib import Path
from typing import List, Optional, Union

from Levenshtein import StringMatcher

keep_summary = [
    "PURPOSE",
    "PROPOSED ACT",
    "CONTENT",
    "BUDGETARY IMPLICATIONS",
    "BUDGETARY CONSTRAINTS",
    "DELEGATED ACTS",
    "IMPLEMENTING ACTS",
]

keep_final_act = [
    "PURPOSE",
    "LEGISLATIVE ACT",
    "CONTENT",
]


def edit_dist(x: str, y: str):
    return StringMatcher.distance(x, y)


def compare(
    test_str: str, candidates: Union[List[str], str], edit_distance: int = 2
) -> Optional[str]:
    if isinstance(candidates, list):
        for candidate in candidates:
            token = compare(test_str, candidate, edit_distance)
            if token is not None:
                return token
        return None
    elif isinstance(candidates, str):
        within_dist = False
        if edit_distance > 0:
            within_dist = edit_dist(test_str, candidates) <= edit_distance
        status = within_dist or candidates in test_str
        if status:
            return candidates
        return None


def parse_document(
    document: str, keep, keep_after_content: bool = False, edit_distance: int = 0
):
    out = []
    splitted = re.split("([A-Z ]+:.)", document)
    for i, split in enumerate(splitted):
        test_split = split.replace(":", "").strip()
        if keep_after_content:
            # if "CONTENT" in split:
            match = compare(test_split, "CONTENT", edit_distance)
            if isinstance(match, str):
                match += ": "
                match += " ".join(splitted[i + 1 :]).strip()
                out.append(match)
                break
        match = compare(test_split, keep, edit_distance)
        if isinstance(match, str):
            match += ": "
            try:
                out.append((match + splitted[i + 1]).strip())
            except IndexError:
                out.append(splitted[i].strip())
    return " ".join(out)


def read_file(path) -> str:
    with open(path, "r") as file:
        out = " ".join([line.strip() for line in file]).strip()
    return out


def write_file(text, path: Path) -> None:
    with open(path, "w") as file:
        # file.write("\n".join(text))
        file.write(text)


def main(path: str = "./data/summaries/raw"):
    i = 0
    summaries = Path(path)
    assert summaries.exists(), "Check that you are pointing to the right folder!"
    years = Path(summaries).glob("*")
    for year in years:
        print(f"Processing {year}")
        codes = year.glob("*")
        for code in codes:
            sum_path = code.joinpath("sum")
            txt_files = sum_path.glob("*.txt")
            for file_path in txt_files:
                if any(
                    x in str(file_path) for x in ["final_act", "legislative_proposal"]
                ):
                    # replace non-breaking space if in there
                    raw_text = read_file(file_path).replace("\xa0", " ")
                    alen = len(raw_text)
                    preprocessed_text = parse_document(
                        raw_text,
                        keep=keep_summary
                        if "legislative_proposal" in str(file_path)
                        else keep_final_act,
                        keep_after_content=True,
                        edit_distance=2,
                    )
                    new_alen = len(preprocessed_text)
                    print(f"Removed {alen-new_alen} chars for {file_path}")
                    # if alen-new_alen < 0:
                    #     import pudb
                    #     pu.db
                    if preprocessed_text == " " or preprocessed_text == "":
                        pass
                    target_path = Path(str(file_path).replace("raw", "preprocessed"))
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    write_file(preprocessed_text, target_path)
                    i += 1
    print(f"Finished filtering summaries! Processed {i} files")


if __name__ == "__main__":
    main()
