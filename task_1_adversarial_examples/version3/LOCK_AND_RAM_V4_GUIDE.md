# Lock & Ram V4 - Guide d'Utilisation

## Après l'Exécution du Job

### 1. Vérifier que le job est terminé

```bash
squeue -u ansart1
```

Si aucun job n'apparaît, le job est terminé.

### 2. Lancer l'analyse API

**Important** : Attendre 15 minutes après la dernière requête API si nécessaire.

```bash
python analyze.py output/submission_ram_v4.npz --mode api
```

Cela génère :
- `logs/analysis_api_YYYYMMDD_HHMMSS.json` : Contient les L2 distances par image
- `logs/api/logits_submission_ram_v4_YYYYMMDD_HHMMSS.json` : Logits bruts de l'API

### 3. Vérifier les résultats

```bash
# Voir le dernier JSON d'analyse
ls -t logs/analysis_api_*.json | head -1 | xargs cat | python -m json.tool | less

# Voir le mapping epsilon utilisé
cat logs/lock_and_ram_v4_epsilon_mapping.json
```

## Itération 3 : Ajustement Bidirectionnel

**Stratégie** : Augmenter les epsilons pour les images FAILED, diminuer légèrement pour optimiser.

### Modifier `lock_and_ram_v4.py`

Dans la fonction `compute_adaptive_epsilons()` (lignes ~150-170) :

**Pour les images qui sont devenues FAILED** :
```python
if not is_success:
    # Augmenter significativement
    new_epsilon = min(prev_epsilon * 1.5, 30.0)
```

**Pour les images SUCCESS avec L2 élevé** :
```python
elif l2_norm > 0.4:  # L2 très élevé - peut essayer de baisser
    new_epsilon = max(prev_epsilon * 0.9, base_epsilon * 0.8)
```

**Pour les images SUCCESS avec L2 bas** :
```python
elif l2_norm < 0.1:  # L2 très bas - peut essayer encore plus bas
    new_epsilon = max(prev_epsilon * 0.95, base_epsilon * 0.9)
```

### Relancer

1. Modifier `run_lock_and_ram_v4.sh` ligne 42 :
   ```bash
   PREVIOUS_SUB="output/submission_ram_v4.npz"
   ```

2. Le script détecte automatiquement le dernier `analysis_api_*.json`

3. Lancer :
   ```bash
   sbatch run_lock_and_ram_v4.sh
   ```

## Itération 4 : Ajustement Unidirectionnel (Finale)

**Stratégie** : Seulement augmenter les epsilons pour garantir le succès. Ne pas diminuer.

### Modifier `lock_and_ram_v4.py`

Dans `compute_adaptive_epsilons()`, remplacer la logique d'optimisation par :

```python
if prev_epsilon is not None:
    if not is_success:
        # Image FAILED - augmenter significativement
        new_epsilon = min(prev_epsilon * 1.5, 30.0)
        change = f"+{new_epsilon - prev_epsilon:.1f}"
    else:
        # Image SUCCESS - garder epsilon actuel ou légèrement augmenter pour sécurité
        # Ne JAMAIS diminuer en itération finale
        if l2_norm > 0.3:  # L2 encore élevé - peut augmenter légèrement
            new_epsilon = min(prev_epsilon * 1.1, base_epsilon * 1.2)
            change = f"+{new_epsilon - prev_epsilon:.1f}"
        else:
            # L2 acceptable - garder epsilon actuel
            new_epsilon = prev_epsilon
            change = "keep"
else:
    # Première itération - utiliser base epsilon
    new_epsilon = base_epsilon
    change = "new"
```

### Relancer

Même procédure que l'itération 3, mais avec la logique modifiée ci-dessus.

## Fichiers Importants

- `logs/lock_and_ram_v4_epsilon_mapping.json` : Mapping epsilon utilisé (chargé automatiquement à l'itération suivante)
- `logs/analysis_api_*.json` : Résultats d'analyse avec L2 par image
- `logs/lock_and_ram_v4_checkpoint.json` : Checkpoint si interruption

## Notes

- **Itération 3** : Peut augmenter ET diminuer les epsilons pour optimisation
- **Itération 4** : Seulement augmenter pour garantir le meilleur score final
- Le système charge automatiquement le mapping epsilon précédent
- Les images SUCCESS sont conservées, seules les L2 sont optimisées

