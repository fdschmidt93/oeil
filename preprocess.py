import re
from pathlib import Path
from typing import List, Optional, Union

from Levenshtein import StringMatcher

# def read_file(path):
#     with open(path, "r") as file:
#         return " ".join(file.readlines()).strip()


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

# def parse_document(
#     document: str, keep, keep_after_content: bool = False, edit_distance: int = 0
# ):
#     out = []
#     splitted = re.split("([A-Z ]+:.)", document)
#     re.split("[A-Z ]+:.", document)
#     for i, split in enumerate(splitted):
#         test_split = split.replace(":", "").strip()
#         if keep_after_content:
#             # if "CONTENT" in split:
#             if compare("CONTENT", test_split, edit_distance):
#                 out.append(" ".join(splitted[i:]).strip())
#                 break
#         if compare(test_split, keep, edit_distance):
#             try:
#                 out.append((splitted[i] + splitted[i + 1]).strip())
#             except IndexError:
#                 out.append(splitted[i].strip())
#     return " ".join(out)

def parse_document(
    document: str, keep, keep_after_content: bool = False, edit_distance: int = 0
):
    out = []
    splitted = re.split("([A-Z ]+:.)", document)
    test_splitted = [x.replace(":", "").strip() for x in splitted]
    for i, split in enumerate(splitted):
        test_split = split.replace(":", "").strip()
        if keep_after_content:
            # if "CONTENT" in split:
            match = compare(test_split, "CONTENT", edit_distance)
            if isinstance(match, str):
                match += ": "
                match += " ".join(splitted[i+1:]).strip()
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

CWD = Path("/work/fabiasch/eia/data")
SEG = CWD.joinpath("filtered")
UNSEG = CWD.joinpath("unsegmented")
UNSEG_YEARS = UNSEG.glob("*")


def read_file(path) -> str:
    with open(path, "r") as file:
        out = " ".join([line.strip() for line in file]).strip()
    return out


def write_file(text, path: Path) -> None:
    with open(path, "w") as file:
        # file.write("\n".join(text))
        file.write(text)


file_path = "../data/unsegmented/2012/0003(COD)/sum/sum_final_act_1.txt"
document = read_file(file_path)
parsed = parse_document(document, keep_final_act, edit_distance=2)


years = UNSEG.glob("*")
for year in years:
    codes = year.glob("*")
    for code in codes:
        sum_path = code.joinpath("sum")
        txt_files = sum_path.glob("*.txt")
        for file_path in txt_files:
            if any(x in str(file_path) for x in ["final_act", "legislative_proposal"]):
                
                # replace non-breaking space if in there
                text = read_file(file_path).replace('\xa0', ' ')
                text = parse_document(
                    text,
                    keep=keep_summary
                    if "legislative_proposal" in str(file_path)
                    else keep_final_act,
                    keep_after_content=True,
                    edit_distance=2
                )
                if text == " " or text == "":
                    pass
                target_path = Path(str(file_path).replace("unsegmented", "filtered2"))
                target_path.parent.mkdir(parents=True, exist_ok=True)
                write_file(text, target_path)
        print(f"Completed {year.suffix}/{code.suffix}")


