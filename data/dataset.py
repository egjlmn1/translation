"""Chunked CSV dataset loader for large translation corpora."""
import random
from functools import partial
from typing import Iterator, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

from tokenizer.base import BaseTokenizer


class TranslationDataset(IterableDataset):
    """
    Iterable dataset that streams a CSV file in chunks.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: BaseTokenizer,
        src_col: str = "en",
        tgt_col: str = "fr",
        max_seq_len: int = 128,
        chunk_size: int = 10_000,
        shuffle_buffer: int = 10_000,
        max_rows: Optional[int] = None,
        skip_rows: int = 0,
        batch_size: int = 32,
        cache: bool = False,
        use_bucketing: bool = False,
    ):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer
        self.max_rows = max_rows
        self.skip_rows = skip_rows
        self.batch_size = batch_size
        self.use_bucketing = use_bucketing

        # Optimization: cache subset if specified or if small enough
        self._subset_cache: Optional[List[Tuple[List[int], List[int]]]] = None
        if self.max_rows and self.max_rows <= 50_000 or cache:
            print(f"[Dataset] Pre-tokenizing {self.max_rows} rows into RAM...")
            self._subset_cache = list(self._iter_from_csv())
            # print(f"[Dataset] Loaded {len(self._subset_cache)} samples")

    def set_cache(self, cache: List[Tuple[List[int], List[int]]]) -> None:
        """Allow pre-loading validation data into RAM once."""
        self._subset_cache = cache

    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        if self._subset_cache:
            # Slice the cache so each worker only sees its assigned portion
            # Example: cache[worker_id::num_workers]
            data = list(self._subset_cache)[worker_id::num_workers]
            
            random.shuffle(data)
            yield from data
            return

        # Partition data for multiple workers
        if worker_info is not None:
            # Each worker skips rows to avoid duplicating work
            # Simple approach: worker 0 takes 1st, 2nd... worker 1 takes 3rd, 4th... 
            # (mod logic inside _iter_from_csv)
            it = self._iter_from_csv(worker_id=worker_id, num_workers=num_workers)
        else:
            it = self._iter_from_csv()

        if self.use_bucketing:
            # Bucketing/Length-based Batching Optimization:
            # We fill a large buffer, sort it by length, and then yield batches.
            # This minimizes padding significantly.
            buffer: List[Tuple[List[int], List[int]]] = []
            for item in it:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer:
                    # 1. Sort by source length to minimize padding
                    buffer.sort(key=lambda x: len(x[0]))
                    
                    # 2. Group into batches to maintain the length-sorting benefit
                    batches = [buffer[i:i + self.batch_size] for i in range(0, len(buffer), self.batch_size)]
                    random.shuffle(batches)
                    for b in batches:
                        yield from b
                    buffer.clear()

            if buffer:
                buffer.sort(key=lambda x: len(x[0]))
                yield from buffer
        else:
            # Simple mode: shuffle buffer, pad to max_seq_len in collate_fn
            buffer: List[Tuple[List[int], List[int]]] = []
            for item in it:
                buffer.append(item)
                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    yield from buffer
                    buffer.clear()

            if buffer:
                random.shuffle(buffer)
                yield from buffer

    def _iter_from_csv(self, worker_id: int = 0, num_workers: int = 1) -> Iterator[Tuple[List[int], List[int]]]:
        rows_read = 0
        skipped = 0
        global_row_index = 0
        vocab_size = self.tokenizer.vocab_size
        pad_id = self.tokenizer.pad_id
        sos_id = self.tokenizer.sos_id
        eos_id = self.tokenizer.eos_id

        reader = pd.read_csv(
            self.csv_path,
            chunksize=self.chunk_size,
            usecols=[self.src_col, self.tgt_col],
            skiprows=range(1, self.skip_rows + 1) if self.skip_rows > 0 else None,
            on_bad_lines="skip",
            encoding="utf-8",
        )

        for chunk in reader:
            chunk = chunk.dropna(subset=[self.src_col, self.tgt_col])
            # 1. First, partition the RAW strings (very fast)
            # This ensures we only tokenize what this worker actually needs
            chunk_indices = [i for i in range(len(chunk)) if (global_row_index + i) % num_workers == worker_id]
            global_row_index += len(chunk)
            
            if not chunk_indices:
                continue
                
            chunk_partition = chunk.iloc[chunk_indices]
            
            # 2. Vectorized text extraction on partition
            src_texts = chunk_partition[self.src_col].astype(str).tolist()
            tgt_texts = chunk_partition[self.tgt_col].astype(str).tolist()
            
            # 3. Parallel tokenization (only on assigned rows)
            src_token_batches = self.tokenizer.encode_batch(src_texts)
            tgt_token_batches = self.tokenizer.encode_batch(tgt_texts)
            
            for src_ids, tgt_ids in zip(src_token_batches, tgt_token_batches):
                # 1. Filter by max length — total tokens (incl. SOS + EOS) must be < max_seq_len
                if len(src_ids) >= self.max_seq_len or len(tgt_ids) >= self.max_seq_len:
                    skipped += 1
                    continue
                
                # 2. Filter by min length 
                # (Need at least 2 tokens because teacher forcing does tgt[:, :-1])
                if len(src_ids) < 2 or len(tgt_ids) < 2:
                    skipped += 1
                    continue

                # 3. Out-of-bounds token ID check
                # Any ID outside [0, vocab_size) will cause an illegal memory access
                # in nn.Embedding on CUDA, surfacing as "illegal instruction"
                if any(tid < 0 or tid >= vocab_size for tid in src_ids):
                    skipped += 1
                    continue
                if any(tid < 0 or tid >= vocab_size for tid in tgt_ids):
                    skipped += 1
                    continue

                # 4. All-pad check — sequences of only pad tokens cause NaN in attention
                if all(tid == pad_id for tid in src_ids):
                    skipped += 1
                    continue
                if all(tid == pad_id for tid in tgt_ids):
                    skipped += 1
                    continue

                # 5. SOS/EOS structure check — tokenizer should have added these
                if src_ids[0] != sos_id or src_ids[-1] != eos_id:
                    skipped += 1
                    continue
                if tgt_ids[0] != sos_id or tgt_ids[-1] != eos_id:
                    skipped += 1
                    continue

                yield (src_ids, tgt_ids)

                rows_read += 1
                if self.max_rows is not None and rows_read >= self.max_rows:
                    if skipped > 0:
                        print(f"[Dataset] Filtered {skipped} problematic rows")
                    return

        if skipped > 0:
            print(f"[Dataset] Filtered {skipped} problematic rows")

    def __len__(self) -> int:
        if self._subset_cache:
            return len(self._subset_cache)
        if self.max_rows:
            return self.max_rows
        return 0


def collate_fn(
    batch: List[Tuple[List[int], List[int]]],
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic padding: pad to the longest sequence in the batch."""
    src_batch, tgt_batch = zip(*batch)
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)
    src_padded = [s + [pad_id] * (src_max_len - len(s)) for s in src_batch]
    tgt_padded = [t + [pad_id] * (tgt_max_len - len(t)) for t in tgt_batch]
    return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)


def collate_fn_fixed(
    batch: List[Tuple[List[int], List[int]]],
    pad_id: int = 0,
    max_seq_len: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed padding: pad every sequence to max_seq_len."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = [s + [pad_id] * (max_seq_len - len(s)) for s in src_batch]
    tgt_padded = [t + [pad_id] * (max_seq_len - len(t)) for t in tgt_batch]
    return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)


def create_dataloaders(
    csv_path: str,
    tokenizer: BaseTokenizer,
    batch_size: int = 64,
    max_seq_len: int = 128,
    chunk_size: int = 10_000,
    shuffle_buffer: int = 10_000,
    max_rows: Optional[int] = None,
    val_split: float = 0.05,
    num_workers: int = 0,
    src_col: str = "en",
    tgt_col: str = "fr",
    use_bucketing: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if max_rows:
        total = max_rows
    else:
        import os
        file_size = os.path.getsize(csv_path)
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(1_000_000)
        lines_in_sample = sample.count("\n")
        bytes_per_line = 1_000_000 / max(lines_in_sample, 1)
        total = int(file_size / bytes_per_line)

    pad_id = tokenizer.pad_id
    
    val_rows = int(total * val_split) if val_split > 0 else 0
    # Safety: don't load 2 million rows into RAM for validation if on a huge dataset.
    # 10,000 rows is more than enough for a stable BLEU score.
    val_rows = min(val_rows, 10_000) 
    train_rows = total - val_rows
    # train_rows = 2_000_000
    # train_rows = 100_000

    # 1. Create Train Dataset (Streaming)
    # Use skip_rows=val_rows to skip the validation part
    train_ds = TranslationDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        src_col=src_col,
        tgt_col=tgt_col,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        shuffle_buffer=shuffle_buffer,
        max_rows=train_rows if train_rows else None,
        skip_rows=val_rows,
        batch_size=batch_size,
        # cache=True,
        use_bucketing=use_bucketing,
    )

    # 2. Create Validation Dataset (Cached in RAM)
    val_loader = None
    # We always use the first val_rows for validation
    val_ds = TranslationDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        src_col=src_col,
        tgt_col=tgt_col,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        shuffle_buffer=0,
        max_rows=val_rows,
        # skip_rows=10_000_000,
        skip_rows=0,
        batch_size=batch_size,
        use_bucketing=use_bucketing,
    )

    print(f"[Dataset] Pre-loading validation set ({val_rows} rows) from start of file...")
    val_cache = list(val_ds._iter_from_csv())
    # If file is smaller than 10k, it will just load what's there
    if not val_cache:
        print("[Dataset] WARNING: Validation set is empty!")
    val_ds.set_cache(val_cache)

    # Choose collate function based on bucketing mode
    if use_bucketing:
        train_collate = partial(collate_fn, pad_id=pad_id)
        val_collate = partial(collate_fn, pad_id=pad_id)
    else:
        train_collate = partial(collate_fn_fixed, pad_id=pad_id, max_seq_len=max_seq_len)
        val_collate = partial(collate_fn_fixed, pad_id=pad_id, max_seq_len=max_seq_len)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=val_collate,
        num_workers=0,
        persistent_workers=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=train_collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
