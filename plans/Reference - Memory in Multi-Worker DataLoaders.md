# Memory in Multi-Worker DataLoaders: fork(), Copy-on-Write, and Python Reference Counting

This document explains the interaction between Linux process forking, CPython's memory model, and PyTorch's DataLoader that causes unexpected memory growth with `num_workers > 0` — and why `persistent_workers=True` makes it dramatically worse. Understanding this is essential before enabling multi-worker data loading in training.

## 1. The Surface-Level Symptom

[PyTorch issue #62066](https://github.com/pytorch/pytorch/issues/62066), reported in July 2021, shows that using `DataLoader` with `persistent_workers=True` and `num_workers > 0` causes RAM to climb across epochs — in the reproduction case, from ~14 GB to ~47 GB over 9 epochs. Setting `persistent_workers=False` keeps memory stable at ~14.5 GB. The reproducer is deceptively simple: a dataset holding a large NumPy array, iterated with a multi-worker persistent DataLoader. Nothing obviously wrong with the code.

This is a specific manifestation of the older and more fundamental [issue #13246](https://github.com/pytorch/pytorch/issues/13246), reported as early as 2018 and labelled as a "dependency bug" — the root cause is not in PyTorch's code but in the interaction between CPython's memory model and Linux's `fork()` semantics.

## 2. The Three Systems You Need to Hold in Your Head

Understanding this bug requires knowing about three systems simultaneously. Each behaves reasonably in isolation. The emergent interaction is what causes the problem.

### 2.1 Linux: `fork()` and Copy-on-Write

When PyTorch's DataLoader spawns worker processes (using Python's `multiprocessing` with `start_method="fork"`, the Linux default), the child process gets a virtual copy of the parent's entire address space. Linux uses a **copy-on-write (CoW)** mechanism: the child shares the parent's physical memory pages, and only duplicates a page when either process **writes** to it.

In theory, if workers only *read* the dataset, no pages should be copied and memory stays shared. A 4 GB dataset with 4 workers should still use ~4 GB of physical RAM total.

### 2.2 CPython: Reference Counting

Every Python object has a reference count (`ob_refcnt`) stored in the object's header — a small integer that tracks how many references point to it. Crucially, **accessing** any Python object — even "just reading" it — **increments** that refcount:

```python
x = some_list[idx]  # Increments refcount on some_list AND on the element at idx
```

When `x` goes out of scope, the refcount is decremented. These refcount mutations are **writes to the memory page containing the object header**. From the OS's perspective, the worker process has written to a shared page, triggering a CoW page copy.

This turns copy-on-write into what is effectively **copy-on-read**. The worker thinks it's reading; the OS sees a write.

### 2.3 PyTorch: DataLoader Worker Architecture

With `num_workers > 0`, the DataLoader forks worker subprocesses that each hold their own copy of the `Dataset` object (initially shared via CoW). Workers receive sample indices from the main process, call `dataset[idx]`, and send results back via a queue. With `persistent_workers=False`, workers are killed and re-forked each epoch. With `persistent_workers=True`, they survive indefinitely.

## 3. The Chain of Causation

Here's what happens step by step:

1. **Parent process** creates a dataset holding data in memory (e.g., a large NumPy array, a list of file paths, a HuggingFace Arrow table).

2. **Workers are forked.** Each worker's virtual address space points to the same physical memory pages as the parent. Physical memory usage is still ~1x the dataset size.

3. **A worker executes `self.data[idx]`.** Python's machinery:
   - Looks up `self` → touches refcount on the dataset object
   - Looks up `.data` → touches refcount on the data attribute
   - Indexes with `[idx]` → touches refcount on the element (or the array's internal structures)
   - Each refcount touch is a write → each triggers a CoW page copy

4. **With `shuffle=True`**, workers access samples scattered across the entire dataset over the course of an epoch. This means refcount writes eventually touch **every memory page** that holds dataset elements. Every page gets copied into every worker's private memory.

5. **Physical memory approaches `dataset_size × num_workers`.** In the original bug: ~4 GB × ~8 workers ≈ the observed ~46 GB.

## 4. Why `persistent_workers` Makes It Dramatically Worse

**Without persistent workers (`persistent_workers=False`):**
Workers are killed at the end of each epoch. Their private (CoW-duplicated) pages are freed by the OS. New workers are forked from the parent, starting again with a fresh shared view of memory. Memory spikes during the epoch but drops back between epochs.

**With persistent workers (`persistent_workers=True`):**
Workers are never killed. The CoW-duplicated pages accumulate across epochs and are **never reclaimed**. Each epoch causes workers to walk through more of the dataset, touching more pages, and since the workers are never killed, the duplicated pages only grow. Memory ratchets upward epoch after epoch until it plateaus once all pages have been copied — at which point you're at the full `dataset_size × num_workers`.

The ratcheting pattern is visible in the original bug report: memory grows in bursts across epochs 3–5 as different pages get touched for the first time, then plateaus around epoch 6–7 once most pages have been duplicated.

## 5. Why the Data Structure Matters Enormously

### Python Lists: The Worst Case

A Python list of 10,000 items is 10,000 separate Python objects, each with its own refcount, potentially on different memory pages. Accessing any element touches that element's refcount, dirtying its page. With shuffled access over a full epoch, every element gets touched, and every page gets duplicated.

### NumPy Arrays with Native dtypes: Much Better

A NumPy array with a native dtype (e.g., `float64`, `int32`) is a single Python object wrapping a contiguous C-allocated data buffer. The array object itself has one refcount, but the data buffer has no per-element refcounts. Indexing `arr[idx]` creates a new Python object for the result but **reads from the data buffer without modifying it** — no refcount writes to the data pages. Only the array header page gets CoW-copied (one page, negligible).

**Caveat:** NumPy arrays with `dtype=object` behave like Python lists — each element is still a separate Python object with its own refcount. The fix only works for native numeric dtypes where data is stored as raw bytes.

### NumPy Serialization Trick

[Yuxin Wu's blog post](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/) demonstrates a technique that reduces a COCO dataset from 10 GB total PSS (with 4 workers) to 1.6 GB:

```python
# Serialize each sample to bytes, pack into a single contiguous array
lst = [np.frombuffer(pickle.dumps(x), dtype=np.uint8) for x in lst]
_addr = np.cumsum([len(x) for x in lst])
_lst = np.concatenate(lst)  # One giant np.uint8 array — single refcount
```

This collapses thousands of Python objects into two NumPy arrays (the data blob and the offset index). Workers access the raw byte buffer (no refcount touches), then deserialise individual samples on demand.

### PyTorch Tensors: Shared Memory via ForkingPickler

PyTorch customises Python's `ForkingPickler` so that when a `torch.Tensor` is pickled for multiprocessing, the tensor data gets moved to shared memory (under `/dev/shm`) rather than being serialised to bytes. This means that even with `start_method="spawn"` or `"forkserver"` (which don't benefit from `fork()`'s CoW at all), the tensor data is still shared across processes.

NumPy arrays don't get this treatment — they get fully serialised and copied. This makes `torch.Tensor` the preferred container for large data when using `spawn` or `forkserver`.

## 6. Quantitative Comparison

From Yuxin Wu's measurements on a COCO dataset with 4 workers:

| Approach | Total PSS | Per-Worker USS | Notes |
|---|---|---|---|
| Naive `fork()` with Python objects | 10 GB | 1.8 GB | CoW defeated by refcounting |
| NumPy serialization (`fork`) | 1.6 GB | ~4 MB | Bulk data in single array |
| `torch.Tensor` (`spawn`) | 2.2 GB | 160 MB | 160 MB is `import torch` overhead |
| `torch.Tensor` + `forkserver` preload | 1.7 GB | ~17 MB | Preload eliminates import overhead |

Key memory metrics:
- **USS** (Unique Set Size): Private memory exclusive to one process — this is what grows with CoW duplication
- **PSS** (Proportional Set Size): Fair-share accounting that divides shared pages by the number of sharers — the right metric for total system memory usage
- **RSS** (Resident Set Size): Total RAM held by a process (private + shared) — misleading because it double-counts shared pages

## 7. Additional Techniques

### `gc.freeze()`

Python's `gc.freeze()` moves all currently tracked objects out of the garbage collector's generations, preventing the GC from scanning (and thus touching refcounts of) those objects. This can reduce CoW page dirtying from GC cycles, but doesn't prevent refcount touches from normal attribute access in `__getitem__`. It's a partial mitigation, not a fix.

### `forkserver` with Preloading

Each freshly spawned worker must independently `import torch`, consuming ~160 MB of private memory. Using the (largely undocumented) `multiprocessing.set_forkserver_preload(["torch"])` API preloads torch in the forkserver process before it forks workers, so the import overhead is shared via CoW.

### Multi-GPU Dataset Sharing

In distributed training (one process per GPU), naive approaches replicate the dataset N times (once per GPU process). The fix is to have GPU 0 load and serialise the data into a `torch.Tensor` in shared memory, then distribute `ForkingPickler`-serialised handles to other GPU processes. This prevents N-fold replication.

## 8. The CPython Root Cause

[CPython issue #84436](https://github.com/python/cpython/issues/84436) tracks the upstream effort: "Fixing Copy on Writes from reference counting and immortal objects." Python 3.12 introduced PEP 683 (immortal objects) which marks certain built-in constants and interned strings as having a refcount that is never modified. But this only applies to a small set of built-in objects, not arbitrary user data. PEP 703 (removing the GIL) also reworks refcounting in ways that may eventually help, but as of Python 3.13, the free-threaded build is experimental and not used in typical PyTorch training environments.

The fundamental tension remains: CPython's reference counting provides deterministic, immediate memory management, but it makes `fork()`-based memory sharing unreliable for any workload that touches Python objects.

## 9. Implications for This Codebase

Our datasets use HuggingFace `load_dataset()`, which stores data in memory-mapped Apache Arrow files. Arrow data is read via `mmap()` — the kernel manages the page cache, and reads don't touch Python refcounts on the bulk data. This makes us **largely immune** to the CoW explosion for the actual sample data.

However, the following would still be subject to CoW:
- The `datasets.Dataset` Python wrapper objects and their internal index structures
- Any Python-level caches or metadata dictionaries
- Temporary Python objects created during `__getitem__` processing (prompt construction, tokenization intermediates)

Given this, the current defaults (`num_workers=0`, `persistent_workers=False`) are safe. When moving to `num_workers > 0`:

1. **Start with `persistent_workers=False`** — accept the per-epoch worker respawn cost, get clean memory semantics.
2. **Monitor PSS** (not RSS) to measure actual memory impact — `psutil.Process(pid).memory_full_info().pss` or read `/proc/{pid}/smaps`.
3. **Only enable `persistent_workers=True`** after confirming that memory growth is acceptable for your dataset size and worker count.
4. **Avoid storing data as Python lists or dicts** in the dataset. The Arrow-backed storage already handles this correctly; the risk is in any custom metadata structures.

## 10. References

- [PyTorch #62066 — Memory Leak Found in Persistent DataLoader](https://github.com/pytorch/pytorch/issues/62066) — the persistent workers reproduction
- [PyTorch #13246 — DataLoader num_workers > 0 causes memory replication](https://github.com/pytorch/pytorch/issues/13246) — the canonical upstream root cause (100+ comments, community workarounds)
- [Yuxin Wu — Demystify RAM Usage in Multi-Process DataLoader (2022)](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/) — the best technical deep-dive with quantitative measurements
- [CPython #84436 — Fixing Copy on Writes from reference counting](https://github.com/python/cpython/issues/84436) — the upstream CPython effort
- [PyTorch Data docs — torch.utils.data](https://pytorch.org/docs/stable/data.html) — official documentation acknowledging the issue
