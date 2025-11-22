# How run_pgd.py Works (Concise)

## Goal
Generate adversarial images that:
- Misclassify on the (unknown) target model
- Stay visually close to originals (small L2 distance)

We attack a local surrogate model (ResNet18) and rely on transferability.

## Pipeline
1) Load dataset (`natural_images.pt`): tensors in [0,1], shape (N, 3, 28, 28)
2) Surrogate model: `torchvision.models.resnet18(weights='IMAGENET1K_V1')`
3) Resize to 224×224 and normalize with ImageNet stats
4) Run PGD (Projected Gradient Descent)
5) Save `.npz` with `images` and `image_ids`

## PGD Details
- Objective (untargeted): maximize CrossEntropy(model(x_adv), true_label)
- Update rule:
  - Compute gradient ∇x of loss w.r.t. image
  - Take a step in the sign of the gradient (pixel-wise)
    - `x_adv ← x_adv + α · sign(∇x)`
  - Project back to an L2 ball around the original:
    - `δ = x_adv − x_orig`
    - `if ||δ||2 > ε: δ ← δ · (ε / ||δ||2)`
    - `x_adv ← x_orig + δ`
  - Clip to valid pixel range [0,1]

Notes:
- The step is NOT uniform noise. It is aligned with the gradient’s sign PER PIXEL, i.e., pixels that increase loss are nudged; others are reduced.
- We use an L∞-like step (sign of grad) with an L2 projection (standard and effective in practice).

## Why Loss Increases
If loss rises each step (e.g., 9 → 70), the surrogate is getting more “confused” (higher confidence in a wrong class). That’s expected.

## Why Score Is High Now
- With ε=10, average L2≈10 (raw), which is huge.
- The leaderboard normalizes L2 to [0,1]. Large ε → large normalized distance → bad score, even if misclassified.
- We need to keep misclassification high while driving ε down.

## Surrogate Model Choice
- Current: ResNet18 (fast, standard)
- Better transfer: Use an ensemble (e.g., ResNet50 + DenseNet121 + VGG16_BN) and average the loss/gradients. This often reduces the ε needed to transfer.
- “Best classifier” ≠ “best surrogate.” Transfer improves with architectural diversity, not just accuracy.

## Next Steps (Recommended)
1) Auto-tune ε (surrogate-only, no API calls): binary search per image to find smallest ε that still fools the surrogate.
2) Add ensemble surrogate (ResNet50, DenseNet121, VGG16_BN) and average gradients.
3) Try input diversity (random scaling/padding) to improve transfer.
4) Increase steps to 100–200, reduce α accordingly.

## Estimating Leaderboard Score Locally
`analyze_submission.py` computes:
- Success Rate from API logits (predicted ≠ true)
- Normalized L2 per image:
  - `l2_norm = ||x_adv − x_orig||2 / sqrt(3·28·28)`
  - Misclassified → use `l2_norm`
  - Correctly classified → use `1.0`
- Final score = mean over 100 images (closer to 0 is better)


