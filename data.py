import os, json, torch
from torch.utils.data import Dataset
from functools import lru_cache

class FeatCaptionDataset(Dataset):
    """
    Each item:
      {
        "patch_tokens": FloatTensor [T, D],  # fp32
        "input_ids":   LongTensor   [L],
        "attention_mask": LongTensor [L],
        "labels":      LongTensor   [L]  (-100 on pads)
      }
    """
    def __init__(self, jsonl_path, tokenizer, max_len=64, shard_cache_size=2):
        assert os.path.isfile(jsonl_path), f"Missing {jsonl_path}"
        self.tok = tokenizer
        self.max_len = int(max_len)

        if self.tok.pad_token_id is None:
            if self.tok.eos_token_id is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                self.tok.add_special_tokens({"pad_token": "<|pad|>"})
        self.pad_id = self.tok.pad_token_id

        self.items, self.mode = [], None
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                s = ln.strip()
                if not s:
                    continue 
                try:
                    row = json.loads(s)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSON at line {i} in {jsonl_path}: {e}") from e
                if self.mode is None:
                    self.mode = (
                        "per_file" if "feat_path" in row
                        else "sharded" if ("shard" in row and "row" in row)
                        else None
                    )
                    if self.mode is None:
                        raise ValueError("JSONL needs ('feat_path','caption') or ('shard','row','caption').")

                if self.mode == "per_file":
                    self.items.append({
                        "feat_path": row["feat_path"],
                        "caption": row.get("caption", "")
                    })
                else:
                    self.items.append({
                        "shard": row["shard"],
                        "row": int(row["row"]),
                        "caption": row.get("caption", "")
                    })

        if self.mode == "sharded":
            self._load_shard = lru_cache(maxsize=max(1, shard_cache_size))(self._load_shard_uncached)

    def __len__(self):
        return len(self.items)

    def _load_shard_uncached(self, path):  
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shard not found: {path}")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def __getitem__(self, idx):
        r = self.items[idx]

        if self.mode == "per_file":
            if not os.path.exists(r["feat_path"]):
                raise FileNotFoundError(f"Feature file not found: {r['feat_path']}")
            toks = torch.load(r["feat_path"], map_location="cpu")
        else:
            arr = self._load_shard(r["shard"])
            if torch.is_tensor(arr):
                N = arr.size(0)
            else:
                N = len(arr)
            row = int(r["row"])
            if not (0 <= row < N):
                raise IndexError(f"Row {row} out of bounds for shard {r['shard']} (size {N})")
            toks = arr[row]

        if not torch.is_tensor(toks):
            toks = torch.as_tensor(toks)
        toks = toks.float()

        enc = self.tok(
            r["caption"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "patch_tokens": toks,         
            "input_ids": input_ids,       
            "attention_mask": attention_mask, 
            "labels": labels,             
        }
