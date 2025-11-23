# Guide - Itération Finale (V4 Final)

## Objectif
Restaurer epsilon=30 pour les 48 images qui ont échoué en itération 3, afin de garantir 100% de succès comme en itération 2.

## Résultats des itérations précédentes

- **Itération 2** (submission_ram_v3): epsilon=30 fixe → **100% SUCCESS** (Score: 0.187931)
- **Itération 3** (submission_ram_v4): epsilons adaptés (0.8, 6, 8, 20) → **52% SUCCESS, 48% FAILED** (Score: 0.570223)

## Images concernées

48 images FAILED en itération 3 (qui étaient SUCCESS en itération 2):
- IDs: [0, 5, 6, 7, 8, 9, 12, 13, 15, 18, 20, 21, 22, 23, 25, 26, 29, 30, 32, 34, 36, 41, 44, 45, 46, 50, 51, 58, 59, 60, 62, 63, 66, 67, 69, 71, 73, 77, 78, 79, 82, 83, 88, 89, 95, 96, 97, 98]

## Stratégie finale

- **Images FAILED** (48): epsilon = 30.0 (valeur qui avait fonctionné en itération 2)
- **Images SUCCESS** (52): garder epsilon actuel (ou légèrement augmenter pour sécurité)

## Exécution

```bash
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3
sbatch run_lock_and_ram_v4.sh
```

Le script va automatiquement:
1. Charger `output/submission_ram_v4.npz` (itération 3)
2. Charger `logs/analysis_api_20251123_120511.json` (analyse itération 3)
3. Charger `logs/lock_and_ram_v4_epsilon_mapping.json` (epsilons itération 3)
4. Appliquer la stratégie finale: epsilon=30 pour FAILED, garder pour SUCCESS
5. Générer `output/submission_ram_v4_final.npz`

## Vérification après exécution

```bash
# Analyser avec l'API (attendre ~15 min après soumission précédente)
bash /tmp/analyze_v4.sh output/submission_ram_v4_final.npz

# Soumettre au leaderboard
bash /tmp/submit_v4.sh output/submission_ram_v4_final.npz
```

## Paramètres utilisés

- Epsilon: adaptatif (30.0 pour FAILED, actuel pour SUCCESS)
- Kappa: 50.0
- PGD Steps: 80
- Restarts: 5
- GPUs: 2
