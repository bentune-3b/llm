#!/usr/bin/env python3
# clean_dataset.py
# ------------------------
# Clean and deduplicate existing JSONL train/val splits.
# Input files are hard-coded. Outputs are written alongside.
# ------------------------
# Team: Bentune 3b
# Deep Goyal, Namita Shah, Jay Pavuluri, Evan Zhu, Navni Athale

import json,re,unicodedata
from hashlib import md5
from typing import Optional,Dict
from transformers import AutoTokenizer
from tqdm import tqdm
import re

TRAIN_IN="model/train_set.jsonl"
VAL_IN="model/val_set.jsonl"
TRAIN_OUT="model/train_set_cleaned.jsonl"
VAL_OUT="model/val_set_cleaned.jsonl"
MAX_TOK=8192

TOKENIZER=AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B",use_fast=True)
URL_RE=re.compile(r"https?://\S+")
WHITESPACE_RE=re.compile(r"\s+")
PROFANITY_RE=re.compile(r"\b(fuck|sex|sexy|dick\b)",re.I)

def _normalize_text(txt:str)->str:
    if txt==None:
        return ""

    txt=unicodedata.normalize("NFC",txt)
    txt="".join(c for c in txt if not unicodedata.category(c).startswith("C"))
    txt=URL_RE.sub("[URL]",txt)
    return WHITESPACE_RE.sub(" ",txt).strip()

def _num_tokens(txt:str)->int:
    return len(TOKENIZER(txt,add_special_tokens=False).input_ids)

def _length_ok(txt:str,min_toks:int=1,max_toks:int=MAX_TOK)->bool:
    n=_num_tokens(txt)
    return min_toks<=n<=max_toks

def _contains_profanity(txt:str)->bool:
    return bool(PROFANITY_RE.search(txt))

def clean_example(ex:Dict[str,str])->Optional[Dict[str,str]]:
    ex["instruction"]=_normalize_text(ex["instruction"])
    ex["input"]=_normalize_text(ex["input"])
    ex["output"]=_normalize_text(ex["output"])
    if not _length_ok(ex["output"],min_toks=1,max_toks=MAX_TOK):
        return None
    if _contains_profanity(ex["instruction"]+ex["input"]+ex["output"]):
        return None
    return ex

_seen_hashes=set()
def dedup(ex:Dict[str,str])->bool:
    h=md5(f"{ex['instruction']}|||{ex['input']}|||{ex['output']}".encode()).hexdigest()
    if h in _seen_hashes:
        return False
    _seen_hashes.add(h)
    return True

def process_split(in_path:str,out_path:str):
    kept=0
    total=0
    with open(in_path,"r",encoding="utf-8") as fin,open(out_path,"w",encoding="utf-8") as fout:
        for line in tqdm(fin,desc=in_path,unit=" lines"):
            total+=1
            try:
                ex=json.loads(line)
            except:
                continue
            cleaned=clean_example(ex)
            if cleaned and dedup(cleaned):
                fout.write(json.dumps(cleaned,ensure_ascii=False)+"\n")
                kept+=1
    print(f"{in_path}: kept {kept}/{total}")

def main():
    process_split(TRAIN_IN,TRAIN_OUT)
    process_split(VAL_IN,VAL_OUT)

if __name__=="__main__":
    main()