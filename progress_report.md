====================================================
RAPPORT DE PROGRÈS DU MÉMOIRE – MERVE ALGAN
Mis à jour le : 10 avril 2025
====================================================


----------------------------------------------------
****  VUE D'ENSEMBLE DE LA MÉTHODOLOGIE ****
----------------------------------------------------

1. Corpus manuel de paires originales / simplifiées (terminé)
2. Enquête de lisibilité (réalisée, potentiellement à réitérer)
3. Modèle de prédiction du gain de lisibilité (en cours d’expérimentation)

----------------------------------------------------
CORPUS
----------------------------------------------------

1. 250 phrases originales extraites de Wikipedia. 
  - Critères de sélection des phrases :  
    - Une phrase aléatoire par article (Wikipedia-API)
    - Longueur entre 95 et 250 caractères 
  - Métadonnées enregistrées :  
    - Texte de la phrase  
    - Titre de l’article  
    - URL de la page  
    - Catégories Wikipedia associées

2. 250 versions simplifiées manuellement.
- 9 méthodes de simplification appliquées :
   - Réduction de la longueur des phrases par découpage (30)
   - Élimination des informations non essentielles (30)
   - Élimination des parenthèses et incises (30)
   - Transformation des propositions relatives (30)
   - Propositions de formes actives au lieu de passives (25)
   - Propositions de structures positives au lieu de négatives (25)
   - Substitution de mots complexes (25)
   - Simplification des clauses complexes (25)
   - Omission des mots redondants (25)

----------------------------------------------------
PACKAGE `readability` (FR)
----------------------------------------------------

- Version adaptée du package `readability` pour le français
- Comparaison des modifications sur GitHub : (https://github.com/andreasvc/readability/compare/master...mervealgan:readability:master)
- Formules intégrées : `Mesnager`, `REL`, `Kandel-Moles`
- Formules écartées de l’extraction (non adaptées au français) : `Kincaid`, `ARI`, `Coleman-Liau`, `FleschReadingEase`, `GunningFogIndex`, `SMOGIndex`, `DaleChallIndex`, `complex_words_dc`, `paragraphs`
- Variables linguistiques ajoutées : `basicwords_fr`, `tobeverb_fr`, `auxverb_fr`, `conjunction_fr`, `preposition_fr`, `pronoun_fr`, `subordination_fr`, `article_fr`, `interrogative_fr`, `nominalization_fr`  
- Comptage des syllabes basé sur `pyphen` (hyphénation FR) (https://pyphen.org/)
- Résultat : 28 caractéristiques extraites (stockées sous `diff_read`) pour utilisation dans les données d’entrée du modèle : `LIX`, `RIX`, `REL`, `KandelMoles`, `Mesnager`, `characters_per_word`, `syll_per_word`, `words_per_sentence`, `sentences_per_paragraph`, `type_token_ratio`, `directspeech_ratio`, `characters`, `syllables`, `words`, `wordtypes`, `sentences`, `long_words`, `complex_words`, `complex_words_mes`, `tobeverb`, `auxverb`, `conjunction`, `preposition`, `nominalization`, `subordination`, `article`, `pronoun`, `interrogative`


**Info sur les ajouts :**

La formule de Mesnager :

Source : (François, T. (2011). An analysis of a French corpus for readability assessment. (https://cental.uclouvain.be/team/tfrancois/articles/Francois2011-thesis.pdf)).

```python
def Mesnager(complex_words_mes, words, sentences):
    return (2 / 3) * (complex_words_mes / words) * 100 + (1 / 3) * words / sentences
```

→ Formule Mesnager utilise complex_words_mes, c’est-à-dire les mots absents de la liste de basicwords_fr (liste de Catach) :

```python
if token.lower() not in basicwords:
    complex_words_mes += 1
```
→ basicwords_fr est une liste de 3558 mots fréquents français (la liste de Catach)

Source : (CATACH, N. (1985). Les listes orthographiques de base du français. Nathan, Paris) :


```python
# 3558 MFW French; CATACH, N. (1985). Les listes orthographiques de base du français. Nathan, Paris
basicwords_fr = frozenset("""
A À ABANDONNER ABBÉ ABORD ABSENCE ABSOLU ABSOLUMENT ACCENT ACCEPTER ACCIDENT ACCIDENTS 
ACCOMPAGNAIENT ACCOMPAGNAIT ACCOMPAGNER ACCOMPLIR ACCOMPLIT ACCORD ACCORDER ACCORDS 
ACHETER ACHEVER ACTE ACTES ACTION ACTIONS ACTIVITÉ ACTIVITÉS ACTUEL ACTUELLE ACTUELLEMENT 
...
""".lower().split())
```

REL Score : 

Source : Projet READI-LREC22 (https://github.com/nicolashernandez/READI-LREC22/blob/main/readability/stats/common_scores.py) : 

```python
def REL_score(syllables, words, sentences):
	return 207 - 1.015 * (words / sentences) - 73.6 * (syllables / words)
```

Kandel-Moles : 

Source : CRAN: Package koRpus - R Project (https://search.r-project.org/CRAN/refmans/koRpus/html/readability-methods.html) :

```python
def KandelMoles(syllables, words, sentences):
	return 209 - 1.15 * (words / sentences) - 68 * (syllables / words)
```
 
Comptage des syllabes basé sur l’hyphénation française via pyphen (Aussi utilisé dans spacy_syllables)

Source :
(https://pyphen.org/);
(https://spacy.io/universe/project/spacy_syllables)

```python
def count_syllables_fr(word):
	import pyphen
	dic = pyphen.Pyphen(lang='fr')
	# Nombre de syllabes = nombre de segments séparés par un tiret (min 1) :
	return max(1, len(dic.inserted(word).split('-')))
```

----------------------------------------------------
ENQUÊTE
----------------------------------------------------

### 1. Première enquête 

- Plateforme : site web personnalisé (Flask + SQLite)
  - Lien : https://evaluerlisibilite.pythonanywhere.com/  
  - Code source : `survey_app.py` (# TODO à lier)
- Fonctionnement :
  - Affichage aléatoire de 20 paires de phrases (originale + simplifiée)
  - Échelle de lisibilité (1 à 7) :
    1 = Très facile à lire, 2 = Facile à lire, 3 = Assez facile à lire, 4 = Neutre,  
    5 = Assez difficile à lire, 6 = Difficile à lire, 7 = Très difficile à lire
  - Quota : max. 5 votes par phrase (distribution équilibrée)
  - Stockage en base de données avec timestamps
- Participants : 67 participants (natif ou C1 minimum en français)
- Données collectées : 1205 évaluations sur 250 paires
- Résultat :  
  - Moyenne des scores par phrase (MOS)  
  - Gain de lisibilité = score_simplifiée - score_originale
  - Ce gain est utilisé comme variable cible dans le modèle de régression.

- Statut :  
  - Utilisée dans les expérimentations actuelles  
  - Limites : pas d’identifiant unique, pas de vérification du niveau, consignes trop ouvertes

- Considérations pour conserver l’enquête existante :
  - L’information sur le niveau requis (natif ou C1) était indiquée dans l’introduction du site :  
    *“Cette enquête s'adresse aux personnes de langue maternelle française ou ayant un niveau de français C1 ou plus.”*

  - L’affichage aléatoire de 20 paires par session, combiné à une limite de 5 votes maximum par phrase, rend peu probable qu’un même participant annote plusieurs fois les mêmes phrases :

```python
# TODO Add the name of the code
c.execute("SELECT id, sentences, simplified FROM allsents WHERE votes < 5 ORDER BY RANDOM() LIMIT 20")
```

  
### 2. Nouvelle enquête (à décider)

- Améliorations prévues :
  - Identifiant pseudonyme ou unique pour chaque participant
  - Question fermée sur le niveau/langue maternelle
  - Instructions plus claires
  - Échelle à rediscuter (5 ou 7 points)

- Format envisagé : 
  - Option 1 : 1 seule question par paire : 
    "Pensez-vous que la phrase est devenue plus facile ou plus difficile à lire ?"
    - Option 1.a : Échelle 7 points :  (`-3` à `+3`)
    Beaucoup plus facile – Plus facile – Un peu plus facile – Pareil – Un peu plus difficile – Plus difficile – Beaucoup plus difficile
    - Option 1.b : Échelle 5 points:  (`-2` à `+2`)
    Beaucoup plus facile – Plus facile – Pareil – Plus difficile – Beaucoup plus difficile

  - Option 2 : 2 questions séparées (une par phrase) :
    "Veuillez évaluer la lisibilité de chaque phrase ci-dessous."
    - Option 2.a : Échelle 7 points : (`1` à `7`)  
      1 = Très facile, 2 = Facile, 3 = Assez facile, 4 = Moyenne,  
      5 = Assez difficile, 6 = Difficile, 7 = Très difficile  
      → Gain de lisibilité est calculé : simplifiée - originale → plage possible : `-6` à `+6`
    - Option 2.b : Échelle 5 points : (`1` à `5`)  
      1 = Très facile, 2 = Facile, 3 = Moyenne, 4 = Difficile, 5 = Très difficile  
      → Gain de lisibilité : simplifiée - originale → plage : `-4` à `+4`

- Statut :
  - L’enquête ne sera lancée qu’une fois tous les éléments suivants validés :  
    - la décision définitive de la refaire,  
    - la confirmation que le pipeline de modélisation est compatible avec les nouvelles annotations,  
    - la validation de la nouvelle version de l’enquête par Mme Todirascu.


----------------------------------------------------
MODÈLE
----------------------------------------------------

### Objectif :
- Prédire le gain de lisibilité perçu entre une phrase originale et sa version simplifiée.

### Options de données d’entrée :

**diff_emb_camem_mean_pooling**
- 768 dimensions : `embed_diff_0` à `embed_diff_767`  
- Différences entre les vecteurs moyens (mean pooling) des phrases originale et simplifiée, extraits avec CamemBERT (`embedding_simplifiée - embedding_originale`)

**diff_emb_camem_mean_pooling_pca**
- 250 composantes principales : version réduite de `diff_emb_camem_mean_pooling` par PCA.  
- Utilisée pour améliorer la stabilité du modèle et réduire la dimensionnalité sans perte significative d'information.

**diff_emb_camem_att_pooling_pca**  
- Différences entre vecteurs attention-weighted pooling
- Réduction à 250 dimensions par PCA

**diff_emb_camem_max_pooling_pca**  
- Différences entre vecteurs max pooling
- PCA, réduction à 250 dimensions

**diff_read**  
- 28 caractéristiques linguistiques et de lisibilité (package `readability` adapté au FR) (diff : simplifiée - original)
- Exemple :  
  `diff_LIX`, `diff_RIX`, `diff_REL`, `diff_KandelMoles`, `diff_Mesnager`, `diff_long_words`, `diff_complex_words`, `diff_nominalization`, `diff_subordination`, etc.

**diff_coref**
- 3 caractéristiques liées aux chaînes de coréférence extraites avec Coreferee (https://github.com/msg-systems/coreferee) :
`diff_n_pronouns`, `diff_has_coref_chain`, `coref_chain_removed`


### Modèles évalués :
- MLPRegressor
  `hidden_layer_sizes=(64, 32)`, `activation='tanh'`, `solver='adam'`,  
  `alpha=0.5`, `learning_rate_init=0.001`, `max_iter=300`,  
  `early_stopping=True`, `random_state=42`
- MLPRegressor (2)
  `hidden_layer_sizes=(128, 64, 32)`, `activation='relu'`, `solver='adam'`, `alpha=0.001`, `batch_size=32`, `learning_rate='adaptive'`, `learning_rate_init=0.001`, `max_iter=500`, `early_stopping=True`, `validation_fraction=0.2`, `random_state=42`
- RandomForestRegressor
  `n_estimators=100`, `random_state=42`
- XGBRegressor
  `n_estimators=100`, `learning_rate=0.1`, `random_state=42`
- XGBRegressor (2)
  `n_estimators=300`, `learning_rate=0.03`, `max_depth=6`, `min_child_weight=2`, `gamma=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.01`, `reg_lambda=1`, `random_state=42`

### Expérimentations :

Expérimentation 1 :

Données d’entrée : (diff_emb_camem_mean_pooling) + (diff_read)  
(code : `exp_flipped_without_coref.py`) ## TODO

| Modèle         | MAE    | RMSE   | Pearson | Spearman |
|----------------|--------|--------|---------|----------|
| MLP            | 0.5518 | 0.6772 | 0.4896  | 0.4799   |
| Random Forest  | 0.5298 | 0.7252 | 0.3314  | 0.3864   |
| XGBoost        | 0.5624 | 0.7081 | 0.4161  | 0.4398   |

-------

Expérimentation 2 :

Données d’entrée : (diff_emb_mean_pooling) + (diff_read) + (diff_coref)  
(code : `exp_flipped_meanpoolwithoutpca.py`) # TODO

| Modèle         | MAE    | RMSE   | Pearson | Spearman |
|----------------|--------|--------|---------|----------|
| MLP            | 0.5534 | 0.7348 | 0.3163  | 0.2385   |
| Random Forest  | 0.5310 | 0.7313 | 0.3059  | 0.3716   |
| XGBoost        | 0.5624 | 0.7081 | 0.4161  | 0.4398   |

------

Expérimentation 3 :

Données d’entrée : (diff_emb_mean_pooling_pca) + (diff_read) + (diff_coref)  
(code : `exp_flipped_meanpool_pca.py`)

| Modèle         | MAE    | RMSE   | Pearson | Spearman |
|----------------|--------|--------|---------|----------|
| MLP            | 0.5822 | 0.7578 | 0.4512  | 0.4693   |
| Random Forest  | 0.5083 | 0.6893 | 0.4401  | 0.4980   |
| XGBoost        | 0.5714 | 0.7673 | 0.2282  | 0.2761   |

------

Expérimentation 4 :

(diff_emb_camem_att_pooling_pca) + (diff_read) + (diff_coref)

| Modèle         | MAE    | RMSE   | Pearson | Spearman |
|----------------|--------|--------|---------|----------|
| MLP            | 0.5802 | 0.7553 | 0.4544  | 0.4565   |
| Random Forest  | 0.4840 | 0.6744 | 0.4967  | 0.5505   |
| XGBoost        | 0.5551 | 0.7696 | 0.2394  | 0.3407   |

---

Expérimentation 5 : (Meilleurs résultats actuels) ********  

Données d’entrée :   
  
(diff_emb_camem_max_pooling_pca) + (diff_read) + (diff_coref)  

| Modèle        | MAE    | RMSE   | Pearson | Spearman |
|---------------|--------|--------|---------|----------|
| MLP           | 0.5934 | 0.7701 | 0.4440  | 0.4411   |
| MLP (2)       | 0.5599 | 0.6900 | 0.4948  | 0.5408   |
| Random Forest | 0.4745 | 0.6514 | 0.5576  | 0.6164   |
| XGBoost       | 0.4960 | 0.6856 | 0.4416  | 0.5131   |
| XGBoost (2)   | 0.4589 | 0.6237 | 0.6166  | 0.6395   |

--------

Expérimentation 6 :

— Approche ensemble (en cours)



Notes : # TODO

- Utilisation of 9 methods in the evaluation? 
- Thesis pdf versions will be added as the modifications done, to the folder of thesis_versions_pdf.





