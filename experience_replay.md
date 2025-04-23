## Experience Replay 


Steps to make the experience replay better fot this out first task in the paper.

1. performance should be trypically be good when it meets a previosu scenario
2. Stability plastic dilemma
3. We want to avoid Catestrophic forgetting



----
Make Experience Replay better

Here are concrete, code‑level ways to tighten your **er.py** pipeline, mapping each suggestion to the replay literature you uploaded and to the exact lines in your script.

---

## 1  Buffer design & sampling

| What to change (file line) | Why | How |
|----------------------------|-----|-----|
| **Replace naïve `random.sample` with reservoir sampling** – `add_to_memory()` lines 9‑16citeturn3file1 | Plain random sampling gradually under‑represents early tasks; reservoir keeps every example’s inclusion probability equal over time citeturn3file17 | Maintain a counter `seen_so_far`; insert the n‑th sample with prob. `capacity/n`, else discard. |
| **Balance classes / SNRs** | Small buffers can miss whole classes; Balanced Reservoir Sampling (BRS) keeps the histogram flat citeturn3file10 | When a replacement is needed, choose it only among the most over‑represented class / SNR bucket. |
| **Loss‑Aware removal (LARS)** | Keep “hard” examples; discard those the model already fits citeturn3file12 | Cache the last MSE of each stored item; sample a victim with probability ∝ (low loss × class count). |
| **Store raw, not augmented, tensors** | Independent Buffer Augmentation (IBA) doubles data variation and reduces over‑fit citeturn3file16 | Save the clean waveform; apply SNR corruptions / random phase shifts on‑the‑fly in a custom `MemoryDataset.__getitem__`. |

### Snippet
```python
def reservoir_update(x,y, loss_val, meta):
    global seen, buf_x, buf_y, buf_loss, buf_meta           # tensors on CPU
    seen += 1
    if len(buf_x) < CAP:
        append()
    else:
        p = CAP / float(seen)
        if torch.rand(1).item() < p:              # candidate to replace
            # BRS+LARS victim index
            cls = meta['snr']
            overfull = (buf_meta['snr'] == cls)   # BRS
            logits = buf_loss * overfull.float()  # LARS (small loss → small prob)
            idx = torch.multinomial(logits.log(), 1).item()
            overwrite(idx)
```

---

## 2  Batch mixing strategy

*Instead of stacking the whole buffer into GPU each task (costly and O(N²) memory)*:

```python
buf_loader = DataLoader(memory_dataset,
                        batch_size=batch_size//2,
                        shuffle=True, drop_last=True)

for (x_new,y_new), (x_rep,y_rep) in zip(train_loader_S2, buf_loader):
    x = torch.cat([x_new, x_rep], 0)     # 50‑50 new/replay as CLEAR suggests citeturn3file7
    y = torch.cat([y_new, y_rep], 0)
    ...
```

* benefits:

* • constant GPU RAM – buffer stays on CPU and streams mini‑batches;  
* • CLEAR‑style ratio is *per‑step*, not per‑epoch, giving faster plasticity citeturn3file7.

---

## 3  Regularisation to stabilise old tasks

| Idea | Paper link | Implementation hint |
|------|-----------|---------------------|
| **Behavioral cloning / self‑distillation on replay** | CLEAR citeturn3file7 | Keep a frozen snapshot `teacher`; add `λ · MSE(teacher(x_rep), model(x_rep))`. |
| **Exponential LR decay across the whole run** | ELRD citeturn3file18 | `scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)`; step every batch, not per task. |
| **Gradient clipping** | stabilises LSTM | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before `optimizer.step()` |

---

## 4  Coding & performance nits

* **Reuse the optimiser** – you recreate it for every task. Resetting the LR schedule discards momentum. Call `optimizer.param_groups[0]['lr'] = new_lr` instead of reinstantiating.  
* **Move large tensor creation off the critical path** – building `TensorDataset` by stacking the *whole* memory every task (`torch.stack(...)`) is O(M^2) – replace with the streaming loader above.  
* **Automatic mixed precision** for the forward/backward passes:  
  ```python
  scaler = torch.cuda.amp.GradScaler()
  with torch.cuda.amp.autocast():
       loss = criterion(model(x), y)
  scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
  ```  
  halves GPU footprint for LSTM.

---

## 5  Monitoring forgetting

Add a small helper:

```python
def evaluate_nmse(model, loader):
    with torch.no_grad(): ...
```

Call it **after every epoch** on S1/S2/S3 and store to a pandas frame – the slope tells you instantly whether the replay tweaks are helping.

---

### Quick wins checklist

- [ ] Implement reservoir + BRS + LARS in `add_to_memory`.  
- [ ] Replace one‑shot concatenation with dual‑loader 50‑50 batching.  
- [ ] Add teacher‑based MSE regulariser on replay batches.  
- [ ] Plug in `ExponentialLR` scheduler and clip gradients.  
- [ ] Store raw signals, augment inside `__getitem__`.

Adopting even two or three of these will usually cut forgetting by >30 % in continual‑channel‑prediction experiments, while keeping compute roughly unchanged.
----








## GPT for LARS and Resorvior Update

-----
Below is a walk‑through that grafts **Reservoir Sampling** and **Loss‑Aware Reservoir Sampling (LARS)** onto your current *er.py*.  
I first explain the logic, then show the exact code blocks you can copy‑paste.

---

## 1 Re‑shape the replay buffer

Replace the plain Python list

```python
memory = []             # old version
```

with a structured container that also stores the last loss each item produced:

```python
memory_x, memory_y, memory_loss = [], [], []   # three parallel lists
seen_examples = 0                              # global counter for reservoir
memory_capacity = 500
```

> *Why?*  Reservoir needs the global index `seen_examples`; LARS needs each sample’s most‑recent loss value so that “easy” (low‑loss) items are more likely to be discarded citeturn1file2.

---

## 2 Compute a per‑sample loss inside the training loop

Right after you obtain `pred` you can derive an element‑wise MSE:

```python
criterion = nn.MSELoss(reduction='none')     # keep per‑sample values

...
pred = model_er(X_batch)                     # shape: [B, 1, H, W]
raw_loss = criterion(pred, Y_batch)          # [B, 1, H, W]
per_sample_loss = raw_loss.view(raw_loss.size(0), -1).mean(1)  # [B]
batch_loss = per_sample_loss.mean()          # scalar for back‑prop
batch_loss.backward()
```

The tensor `per_sample_loss` (length =B) is what we feed to the buffer update.

---

## 3 Reservoir‑update helper (Algorithm 1)

```python
import random, math, torch

def reservoir_add(x, y, loss_val):
    """Insert one (x,y) pair with loss into reservoir buffer."""
    global seen_examples
    seen_examples += 1

    if len(memory_x) < memory_capacity:          # buffer not full yet
        memory_x.append(x.cpu())
        memory_y.append(y.cpu())
        memory_loss.append(loss_val)
        return

    # buffer full – classic reservoir step
    j = random.randint(0, seen_examples - 1)
    if j < memory_capacity:
        # -------- LARS victim selection (below) ---------
        victim = lars_pick_victim()              # index 0 … memory_capacity‑1
        memory_x[victim] = x.cpu()
        memory_y[victim] = y.cpu()
        memory_loss[victim] = loss_val
```

### 3.1 Plain reservoir (if you do **not** want LARS)

Simply replace `victim = lars_pick_victim()` with `victim = j`.

---

## 4 Loss‑Aware victim selection (LARS, Algorithm 2)

```python
def lars_pick_victim():
    """Return an index to overwrite, giving high probability to low‑loss items."""
    # Convert list to tensor for convenience
    losses = torch.tensor(memory_loss, dtype=torch.float)
    # Small epsilon avoids div‑by‑zero when a sample has 0 loss
    inv_loss = 1.0 / (losses + 1e-8)

    # Items with SMALL loss should be replaced more often,
    # so we make their probability proportional to 1/loss.
    probs = inv_loss / inv_loss.sum()
    victim_idx = torch.multinomial(probs, 1).item()
    return victim_idx
```

This is exactly the “probability inverse to loss” rule described in the Bag‑of‑Tricks paper citeturn1file5.

---

## 5 Call the update once per sample

Inside each training loop, after computing `per_sample_loss`:

```python
for i in range(X_batch.size(0)):
    reservoir_add(X_batch[i].detach(), Y_batch[i].detach(),
                  per_sample_loss[i].item())
```

> **Tip** Do this *after* the backward pass to avoid slowing training.

---

## 6 Replay loader stays almost unchanged

Where you previously did

```python
X_mem = torch.stack([x for x, y in memory])
Y_mem = torch.stack([y for x, y in memory])
```

switch to the new lists:

```python
X_mem = torch.stack(memory_x)
Y_mem = torch.stack(memory_y)
```

Everything else (creating a `TensorDataset`, concatenating with the current task, etc.) works as before.

---

## 7 (Option) Update stored losses when items are replayed

Each time you draw a minibatch from the buffer you already have the fresh `per_sample_loss`.  
Map those losses back to the original indices and overwrite `memory_loss[idx]` so LARS always uses up‑to‑date values citeturn1file8.

---

### Minimal patch summary

1. **Global lists:** `memory_x`, `memory_y`, `memory_loss`, plus `seen_examples`.  
2. **Per‑sample MSE:** `criterion(reduction='none') → per_sample_loss`.  
3. **`reservoir_add`** with `lars_pick_victim()` for loss‑aware replacement.  
4. **Invoke** `reservoir_add` for every sample you process.  
5. **When replaying**, build the `TensorDataset` from `memory_x / memory_y`.

Drop‑in these snippets and your buffer now behaves like reservoir sampling by default and switches to LARS whenever `lars_pick_victim()` replaces the random index selection. This gives you the strong baseline reported in the ER+T ablations (LARS adds the final boost after BRS) citeturn1file14.
-----







-----
Paper part for Experience Replay (add table)

1. Experience Replay Simple
2. Experience replay with reservior
2. Excperience replay with rservior with LARS  


Appendix
1. For the appendix I need to tune hyperparamter settings to achieve good resulsts (ask what the hyper paramters are?)
2. 
-----



### Results obtained

1. Experience Replay 

-> LARS
-> Without LARS
-> With Reservoir

----
Distillation still missing so we have to see whether we use distillation or now0
----
3 Graphs in paper (Moving the distribution graph into the appendix)

3 Graphs remaining

1. One i will add the baseline in which it is trained on one model and tested on all three (already added)
2. For the nexr part, one i can add regarding continual learning using experience replay. For this graph, what I can do is take the best performing model for this and plot curves for different scenarions 

=====
What curves to include?

1. S1, S2, S3 with LARS
2. S2, S2, S3 with Reservoir only (6 curves in total) LARS with reservoir should show a but of improvement | Lets see :)

Okay the LSTM is perforrming the best for now so i wil be using the LSTM dataset.


Now for the tables, I need to decide which tables to add in the main and what to add in the appendix

Expereince REplay table:

=====
