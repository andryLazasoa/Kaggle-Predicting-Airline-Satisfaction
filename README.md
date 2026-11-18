# ✈️ Predicting Airline Satisfaction

Prédiction de la satisfaction des passagers d'une compagnie aérienne à partir de leurs notes de service, de leur profil et de leur trajet.
Projet réalisé pour la compétition **Kaggle Playground Series – Saison 6, Épisode 10**.

![Page de la compétition Kaggle](assets/kaggle_competition.png)

| | |
|---|---|
| **Tâche** | Classification binaire (passager satisfait ou non) |
| **Métrique** | ROC AUC |
| **Modèle** | XGBoost |
| **Meilleur score de validation locale** | **0.95976** (voir [Meilleur score](#-meilleur-score)) |

---

## 📌 Sommaire

1. [La compétition Kaggle](#-la-compétition-kaggle)
2. [Le dataset](#-le-dataset)
3. [Description des variables](#-description-des-variables)
4. [Démarche](#-démarche)
5. [Tests et hypothèses](#-tests-et-hypothèses)
6. [Meilleur score](#-meilleur-score)
7. [Limites et pistes d'amélioration](#-limites-et-pistes-damélioration)
8. [Structure du dépôt et utilisation](#-structure-du-dépôt-et-utilisation)

---

## 🏆 La compétition Kaggle

Les **Playground Series** sont une série de compétitions mensuelles organisées par Kaggle : des jeux de données tabulaires accessibles, pensés pour que la communauté pratique le machine learning. Il n'y a ni points ni médailles, seulement du *swag* pour les meilleurs.

| | |
|---|---|
| **Compétition** | Predicting Airline Satisfaction – Playground Series S6E10 |
| **Lien** | <https://www.kaggle.com/competitions/playground-series-s6e10> |
| **Hôte** | Kaggle |
| **Période** | du 1er octobre 2026 au 1er novembre 2026 |
| **Évaluation** | Aire sous la courbe ROC (*ROC AUC*) entre la probabilité prédite et la cible observée |
| **Tags** | Beginner · Tabular · ROC AUC Score |
| **Participation** (à la date de la capture) | 6 660 inscrits · 3 901 participants · 3 734 équipes · 44 249 soumissions |

Le fichier de soumission attendu contient, pour chaque `id` du jeu de test, la **probabilité** que le passager soit satisfait (et non une classe 0/1, puisque la métrique est le ROC AUC).

> Ce projet a été soumis en *late submission* (après la clôture de la compétition).

---

## 📊 Le dataset

Les données proviennent de la compétition. Leurs variables sont celles du jeu public *Airline Passenger Satisfaction* de Kaggle, dont les données des Playground Series sont dérivées.

| Fichier | Lignes | Colonnes | Remarque |
|---|---|---|---|
| `train.csv` | 699 635 | 22 (+ `id`) | contient la cible `satisfaction` |
| `test.csv` | 299 844 | 21 (+ `id`) | pas de cible, à prédire |

**Qualité des données**
- Aucun doublon.
- Une seule colonne avec des valeurs manquantes : `Arrival Delay in Minutes` (292 lignes dans le train, 130 dans le test). Les lignes concernées sont supprimées du train (→ **699 343 lignes**) ; XGBoost gère nativement les valeurs manquantes du test.

**Quelques chiffres**
- Genre : 50,3 % d'hommes / 49,7 % de femmes.
- Classe : 48,9 % Business · 46,8 % Eco · 4,3 % Eco Plus.
- Le taux de satisfaction varie fortement selon la classe : les passagers **Business** sont nettement plus satisfaits que ceux d'**Eco** et **Eco Plus**.

---

## 🧾 Description des variables

### Cible

| Variable | Description |
|---|---|
| `satisfaction` | Satisfaction du passager (binaire). Encodée en 0/1 par ordre alphabétique : **1 = satisfait**. |

### Profil du passager et du trajet

| Variable | Type | Description |
|---|---|---|
| `Gender` | catégorielle | Genre (Male / Female) |
| `Customer Type` | catégorielle | Type de client (fidèle / non fidèle) |
| `Age` | entier | Âge (7 à 85 ans) |
| `Type of Travel` | catégorielle | Motif du voyage (professionnel / personnel) |
| `Class` | catégorielle | Classe de voyage : Business, Eco, Eco Plus |
| `Flight Distance` | entier | Distance du vol (67 à 4 983 ; 3 474 valeurs distinctes) |
| `Departure Delay in Minutes` | entier | Retard au départ (minutes) |
| `Arrival Delay in Minutes` | décimal | Retard à l'arrivée (minutes) – seule colonne avec des valeurs manquantes |

### Notes de satisfaction par service (échelle de 0 à 5)

Un `0` correspond vraisemblablement à un service non applicable / non noté.

| Variable | Service évalué |
|---|---|
| `Inflight wifi service` | Wifi à bord |
| `Departure/Arrival time convenient` | Commodité des horaires de départ / d'arrivée |
| `Ease of Online booking` | Facilité de réservation en ligne |
| `Gate location` | Emplacement de la porte d'embarquement |
| `Food and drink` | Restauration et boissons |
| `Online boarding` | Embarquement en ligne |
| `Seat comfort` | Confort du siège |
| `Inflight entertainment` | Divertissement à bord |
| `On-board service` | Service à bord |
| `Leg room service` | Espace pour les jambes |
| `Baggage handling` | Gestion des bagages (de 1 à 5) |
| `Checkin service` | Service d'enregistrement |
| `Cleanliness` | Propreté |

### Variables les plus informatives

Le calcul de l'**information mutuelle** (MI) avec la cible donne, dans l'ordre :

| Rang | Variable |
|---|---|
| 1 | `Online boarding` (loin devant les autres) |
| 2 | `Class` |
| 3 | `Inflight wifi service` |
| 4 – 7 | `Flight Distance`, `Inflight entertainment`, `Type of Travel`, `Seat comfort` |
| dernières | `Arrival Delay in Minutes`, `Departure/Arrival time convenient`, `Departure Delay in Minutes`, `Gender` (information quasi nulle) |

La matrice de corrélation montre par ailleurs des groupes de variables très liées entre elles : confort (`Seat comfort`, `Inflight entertainment`, `Cleanliness`, `Food and drink`, corrélations ≈ 0,7–0,8) et réservation (`Inflight wifi service`, `Ease of Online booking`, corrélation ≈ 0,8).

---

## 🔬 Démarche

1. **Chargement et EDA** : dimensions, types, valeurs manquantes, doublons, distributions, relation entre classe et satisfaction.
2. **Nettoyage** : suppression des lignes sans `Arrival Delay in Minutes`.
3. **Analyse** : corrélations et information mutuelle.
4. **Encodage** : variables texte transformées en codes entiers.
5. **Score de référence** : XGBoost par défaut.
6. **Feature engineering** : une série d'hypothèses testées une par une (clusters, PCA, target encoding), chacune comparée au score de référence.
7. **Modèle final** : meilleure combinaison retenue, entraînement sur tout le train, prédiction des probabilités sur le test.

Deux protocoles de validation ont été utilisés (voir les fonctions `score_simple_split` et `score_stratified_split` du notebook) :

| | Round 1 | Round 2 |
|---|---|---|
| Split | aléatoire 75/25 | **stratifié** 75/25 |
| Modèle | XGBoost par défaut | XGBoost `hist` |
| Features dépendant de la cible | – | apprises **uniquement sur le train** (encodage *out-of-fold*) |
| Score de référence | 0.95843 | 0.95837 |

⚠️ Les scores des deux rounds ne sont pas strictement comparables entre eux (split différent).

---

## 🧪 Tests et hypothèses

### Round 1 – score de référence : **0.95843**

| # | Hypothèse | Test | ROC AUC | Écart vs réf. |
|---|---|---|---|---|
| 1 | Regrouper les passagers selon leur profil de notes de service apporte un signal supplémentaire | KMeans (k = 4) sur `On-board service`, `Leg room service`, `Inflight wifi service`, `Checkin service`, `Class` → `Service_cluster` | 0.95850 | +0.00007 |
| 2 | Les distances aux centres de clusters sont plus riches qu'un simple label | Distances aux 6 centroïdes (KMeans k = 6, mêmes variables) | 0.95776 | −0.00067 |
| 3 | Donner le taux de satisfaction moyen par classe aide le modèle | `sat_class` = moyenne de la cible par `Class` | 0.95839 | −0.00004 |
| 4a | Résumer les retards / horaires en un cluster | `Service_cluster` + `Delay_cluster` (KMeans k = 2 sur `Arrival Delay in Minutes`, `Departure/Arrival time convenient`, `Departure Delay in Minutes`) | 0.95854 | +0.00011 |
| 4b | Les variables de retard (information mutuelle quasi nulle) peuvent être supprimées | Même jeu que 4a, sans `Delay_cluster` ni les 3 variables de retard / horaire | 0.95852 | +0.00009 |
| 5 | Condenser les variables de retard avec une PCA | 3 composantes PCA sur les variables de retard, ajoutées à baseline + `Service_cluster` | 0.95840 | −0.00003 |
| 6 | Condenser les variables de confort corrélées avec une PCA | 3 composantes PCA sur `Inflight entertainment`, `Seat comfort`, `Cleanliness`, ajoutées à baseline + `Service_cluster` | 0.95826 | −0.00017 |

**Lecture** : aucune de ces idées n'apporte de gain réellement significatif. Les écarts (de l'ordre de 10⁻⁴) sont du même ordre de grandeur que la variabilité d'un simple changement de split. Supprimer les variables de retard ne dégrade pas le score (4b), ce qui confirme leur faible utilité.

### Round 2 – score de référence : **0.95837**

| # | Hypothèse | Test | ROC AUC | Écart vs réf. |
|---|---|---|---|---|
| 7 | Un *target encoding* lissé de `Flight Distance` capture mieux sa relation avec la cible que la valeur brute | `sat_flight_dis` : moyenne lissée de la cible par valeur de `Flight Distance` (lissage = 20, encodage *out-of-fold* à 5 plis sur le train) | **0.95976** | **+0.00139** |

**Pourquoi le target encoding ?**
- *Le problème* : `Flight Distance` prend 3 474 valeurs distinctes, et sa relation avec la satisfaction n'est ni linéaire ni régulière (information mutuelle élevée, 4ᵉ rang).
- *Le besoin* : exprimer « quel est le taux de satisfaction typique pour cette distance ? » sans exploser le nombre de colonnes (un one-hot de 3 474 colonnes est inenvisageable) et sans que le modèle fuite la cible.
- *La solution* : remplacer chaque distance par la moyenne de la cible associée, **lissée** vers la moyenne globale quand une valeur est rare (`(somme + prior × 20) / (effectif + 20)`), et calculée en **out-of-fold** sur le train pour éviter que chaque ligne « voie » sa propre cible.

---

## 🥇 Meilleur score

| | ROC AUC |
|---|---|
| Score de référence (round 2) | 0.95837 |
| **Meilleur score – XGBoost + target encoding de `Flight Distance`** | **0.95976** |
| Score Kaggle (late submission) | _à compléter_ |

Le **meilleur score de validation locale du projet est 0.95976**, obtenu avec le target encoding de `Flight Distance` (+0.0014 par rapport à la référence).

**Modèle utilisé pour la soumission** : `XGBClassifier(random_state=1, tree_method="hist")` entraîné sur l'ensemble du train avec les variables d'origine + `Service_cluster` + `sat_flight_dis`.

---

## ⚠️ Limites et pistes d'amélioration

- **Jointures non alignées (tests 2, 5 et 6)** : les distances aux centroïdes et les composantes PCA ont un index `0..n-1`, alors que le train est indexé par `id` avec des trous après le `dropna`. Ces features sont donc décalées par rapport aux lignes réelles, et les scores de ces trois tests sont à interpréter avec prudence. Piste : réaligner l'index avant la jointure.
- **Clustering du test** : le KMeans du test est ajusté sur le test lui-même, donc les numéros de clusters ne coïncident pas forcément avec ceux du train. Piste : ajuster le KMeans sur le train, puis appliquer `predict` au test.
- **Fuite légère (test 3)** : `sat_class` est calculée sur tout le train avant le split. L'effet est minime (3 catégories seulement) mais le target encoding du round 2 est la méthode correcte.
- **Combinaison finale non évaluée telle quelle** : le score de 0.95976 est mesuré sans `Service_cluster`, alors que le modèle soumis l'inclut.
- **Une seule validation** : un seul split et une seule graine. Une validation croisée donnerait des écarts plus fiables, surtout pour des gains de l'ordre de 10⁻⁴.
- **Autres pistes** : réglage des hyperparamètres (XGBoost / LightGBM), target encoding d'autres variables, ensembling.

---

## 🗂️ Structure du dépôt et utilisation

```
.
├── airline_satisfaction.ipynb   # notebook complet (EDA, expériences, modèle final)
├── README.md
├── assets/
│   └── kaggle_competition.png   # capture de la page de la compétition
└── data/                        # à créer : train.csv et test.csv (téléchargés depuis Kaggle)
```

**Dépendances** : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `jupyter`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

**Exécution**
1. Télécharger `train.csv` et `test.csv` depuis la page de la compétition et les placer dans `data/`.
2. Lancer `jupyter notebook airline_satisfaction.ipynb` et exécuter toutes les cellules.
3. Le fichier `submission.csv` (colonnes `id`, `satisfaction`) est généré à la racine du projet.
