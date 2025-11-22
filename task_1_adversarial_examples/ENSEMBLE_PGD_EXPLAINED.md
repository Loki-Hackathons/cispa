# Ensemble PGD Attack - Complete Explanation

## Overview
This document explains `run_pgd_ensemble.py` and `run_pgd_ensemble_fast.py` in simple terms for someone new to Machine Learning.

---

## What is PGD (Projected Gradient Descent)?

**PGD** is an attack method that creates "adversarial examples" - images that look normal to humans but fool AI models.

**Simple Analogy:** Imagine you're trying to confuse a security guard (the AI model) by wearing a disguise. PGD is like gradually adjusting your disguise based on feedback, making tiny changes until the guard mistakes you for someone else.

---

## The Parameters Explained

### 1. **EPSILON (ε)** - The "Budget" for Changes
- **What it is:** Maximum allowed distance between original and modified image.
- **Think of it as:** How much you're allowed to "mess up" the image.
- **Lower = Better:** Smaller epsilon means less visible changes (better score).
- **Lower = Harder:** Smaller epsilon makes it harder to fool the model.

**Example:**
- `EPSILON = 10.0` → You can make BIG changes (image looks noisy, score is bad).
- `EPSILON = 0.5` → You can make TINY changes (image looks normal, score is good IF it still fools the model).

**In the code:** This is the L2 distance constraint. We project the perturbation back into an epsilon-ball.

---

### 2. **ALPHA (α)** - Step Size
- **What it is:** How big each "step" is when modifying the image.
- **Think of it as:** Walking speed - small steps = careful, big steps = fast but might overshoot.
- **Typical value:** `0.02` to `0.05`
- **Why smaller for smaller epsilon:** When epsilon is small, you need fine control.

**Example:**
- `ALPHA = 0.1` → Big steps, might overshoot the target.
- `ALPHA = 0.01` → Tiny steps, very precise but slow.

---

### 3. **STEPS** - Number of Iterations
- **What it is:** How many times we repeat the attack process.
- **Think of it as:** Number of attempts to improve the disguise.
- **More steps = Better:** Usually 50-200 steps gives better results.
- **More steps = Slower:** Each step requires running the model.

**Example:**
- `STEPS = 20` → Quick but might not be strong enough.
- `STEPS = 100` → Stronger attack, takes longer.

---

### 4. **BINARY_SEARCH_STEPS** (Only in slow version)
- **What it is:** Reduced steps used during binary search (to speed it up).
- **Why:** Binary search needs to test many epsilon values quickly, so we use fewer steps.
- **Typical value:** `20` (vs full `100` steps).

---

## How PGD Works (Step-by-Step)

### Step 1: Start with Original Image
```
Original Image → [Panda, 28x28 pixels]
```

### Step 2: Forward Pass (What does the model think?)
```
Image → Model → Output: "This is a Panda (confidence: 95%)"
```

### Step 3: Calculate Loss (How "wrong" is the model?)
- We want the model to be WRONG.
- Loss = How confident the model is in the CORRECT answer.
- **Goal:** Maximize this loss (make model confused).

### Step 4: Backward Pass (Which pixels to change?)
```
Loss.backward() → Gradient
```
- **Gradient:** A map showing "which pixels to change and in which direction."
- Like a compass pointing toward "more confusion."

### Step 5: Update Image
```
New Image = Old Image + ALPHA × sign(Gradient)
```
- `sign(Gradient)`: Just the direction (+1 or -1), not the magnitude.
- We move in the direction that increases confusion.

### Step 6: Projection (Stay within budget)
```
If ||New - Original|| > EPSILON:
    Scale down to fit within EPSILON
```
- We check if we exceeded our "budget" (epsilon).
- If yes, we scale down the changes.

### Step 7: Clip to Valid Range
```
Image values must be between [0, 1]
```
- Pixel values can't be negative or > 1.

### Step 8: Repeat
- Go back to Step 2, repeat `STEPS` times.

---

## Ensemble Models - Why Use Multiple?

### The Problem: Transferability
- We attack a **Surrogate Model** (ResNet50) locally.
- But the **Black Box** (competition server) might be a different model.
- **Transferability:** Will attacks that fool ResNet50 also fool the Black Box?

### The Solution: Ensemble
Instead of one model, we use **THREE**:
1. **ResNet50** - Deep, residual connections
2. **DenseNet121** - Dense connections
3. **VGG16_BN** - Classic architecture

**How it works:**
- We average the **losses** from all three models.
- We average the **gradients** from all three models.
- **Result:** Attacks that fool ALL three models are more likely to transfer to ANY model.

**Analogy:** Instead of practicing your disguise on one guard, you practice on three different guards. If you fool all three, you're more likely to fool a fourth guard you've never met.

---

## Input Diversity - Random Scaling/Padding

### What it is:
Before feeding images to the model, we randomly:
- **Scale:** Resize slightly (0.9x to 1.1x)
- **Pad:** Add random padding
- **Crop:** Resize back to 224x224

### Why?
- Models can be "overfitted" to specific image sizes.
- By varying the input, we create attacks that work across different scales.
- **Better transferability** to the Black Box.

**Analogy:** You practice your disguise in different lighting conditions, so it works in any lighting.

---

## Binary Search for Epsilon (Slow Version Only)

### What is Binary Search For?
**Goal:** Find the **minimum epsilon value** needed to successfully fool the model for each individual image.

**Why we need this:**
- Different images have different "difficulty levels" - some are easy to fool, some are hard.
- **Easy images:** Can be fooled with epsilon = 0.3 (tiny changes)
- **Hard images:** Need epsilon = 2.0 (larger changes)
- **Problem:** We don't know which images are easy or hard!

**Why minimum epsilon matters:**
- **Lower epsilon = Better score** (less visible changes = higher quality)
- **Higher epsilon = Worse score** (more visible changes = lower quality)
- We want the **smallest epsilon that still works** for each image.

### What Does Binary Search Search For?
Binary search searches through **epsilon values** in a range (e.g., 0.0 to 3.0).

**What "works" means:**
- After running PGD attack with a given epsilon, we check: Does the model predict the **wrong class**?
- **"Works"** = Model is fooled (predicts wrong class) ✅
- **"Fails"** = Model still predicts correct class ❌

### How Binary Search Works:
For each image, we search through epsilon values:

1. **Start with a range:** epsilon_min = 0.0, epsilon_max = 3.0
2. **Try middle value:** epsilon = 1.5 (middle of range)
3. **Run PGD attack** with epsilon = 1.5 (using fewer steps for speed)
4. **Check result:**
   - **If attack succeeds** (model fooled) → Try smaller epsilon (epsilon_max = 1.5)
   - **If attack fails** (model not fooled) → Try larger epsilon (epsilon_min = 1.5)
5. **Repeat** ~5 times, narrowing the range each time
6. **Result:** Find the minimum epsilon that successfully fools the model

**Example:**
```
Image 1: Try epsilon = 1.5 → Works! → Try 0.75 → Works! → Try 0.375 → Fails → Try 0.562 → Works!
Final epsilon for Image 1 = 0.562 (minimum that works)

Image 2: Try epsilon = 1.5 → Fails → Try 2.25 → Works! → Try 1.875 → Works! → Try 1.687 → Fails
Final epsilon for Image 2 = 1.875 (minimum that works)
```

**Result:** Each image gets its **personalized optimal epsilon** (the smallest value that still fools the model).

**Why it's slow:**
- 100 images × 5 binary search iterations × 20 steps per test = 10,000 quick attacks
- Then 100 images × 100 full steps per image = 10,000 more attacks
- Total: 20,000 attacks × 3 models = 60,000 model evaluations!

---

## Fast Version (Recommended)

**`run_pgd_ensemble_fast.py`** skips binary search:
- Uses **one fixed epsilon** for all images (e.g., `EPSILON = 2.0`)
- Processes all 100 images in **one batch**
- Much faster: ~100 steps × 3 models = 300 model evaluations total

**Trade-off:**
- Slightly less optimal (some images might use more noise than needed)
- But 200x faster!

---

## Normalization & Upsampling

### Upsampling (28x28 → 224x224):
- Our images are tiny (28×28 pixels).
- Models expect big images (224×224).
- We **upsample** (stretch) the image.

### Normalization:
- Models were trained on ImageNet data with specific statistics.
- We normalize: `(pixel - mean) / std`
- Mean = `[0.485, 0.456, 0.406]` (RGB)
- Std = `[0.229, 0.224, 0.225]` (RGB)

**Why:** Models expect this format, so we adapt our images.

---

## Summary: What Happens When You Run It?

1. **Load 3 models** (ResNet50, DenseNet121, VGG16_BN)
2. **Load 100 images** from `natural_images.pt`
3. **For each of 100 steps:**
   - Resize images to 224×224
   - Apply random scaling/padding (input diversity)
   - Normalize
   - Run through all 3 models → get average loss
   - Calculate gradient (which pixels to change)
   - Update image pixels
   - Project back to epsilon-ball
   - Clip to [0,1]
4. **Save** modified images to `submission_pgd.npz`

**Time:** ~5-10 minutes on GPU (fast version) vs hours (slow version with binary search).

---

## How to Use

### Fast Version (Recommended):
```bash
sbatch run_attack.sh  # Runs on GPU node
```

### Or manually:
```bash
module load GCC CUDA PyTorch torchvision
python run_pgd_ensemble_fast.py
python analyze_submission.py --mode local  # Check score locally
```

### Tuning Epsilon:
- Start with `EPSILON = 2.0`
- If success rate < 100% → Increase to 3.0
- If success rate = 100% → Decrease to 1.5, then 1.0, etc.
- Goal: Find smallest epsilon that still gives 100% success rate.

---

## Key Takeaways

1. **Epsilon:** Budget for changes (lower = better score, harder attack)
2. **Alpha:** Step size (smaller = more precise)
3. **Steps:** Iterations (more = stronger attack)
4. **Ensemble:** Use multiple models for better transferability
5. **Input Diversity:** Random scaling improves transferability
6. **Binary Search:** Finds optimal epsilon per image (slow but optimal)
7. **Fast Version:** Uses fixed epsilon (fast but slightly suboptimal)

