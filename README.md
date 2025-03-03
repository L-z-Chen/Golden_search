
## Run Optimization

### Step 1: Define the Search Space  
Define the hyperparameters you want to search using **Golden Search**:  

```python
intervals = [
    [1e-5, 1e-2],   # Learning rate (lr)
    [1e-5, 3e-1],   # Weight decay (wd)
    [0.8, 0.99],    # Beta1
    [0.9, 0.999]    # Beta2
]

### Step 2: Execute Optimization  
Ensure the optimizer is updated in `@train_qwen` before running the optimization process.  
```  
### Launch it

```bash
python main.py
```
