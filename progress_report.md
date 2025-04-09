====================================================
RAPPORT DE PROGRÈS DU MÉMOIRE – MERVE ALGAN
Mis à jour le : 8 avril 2025
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


----------------------------------------------------
PACKAGE `readability` (FR)
----------------------------------------------------

- Version adaptée du package `readability` pour le français
- Comparaison des modifications sur GitHub : (https://github.com/andreasvc/readability/compare/master...mervealgan:readability:master)
- Formules intégrées : `Mesnager`, `REL`, `Kandel-Moles`
- Formules écartées de l’extraction (non adaptées au français) : `Kincaid`, `ARI`, `Coleman-Liau`, `FleschReadingEase`, `GunningFogIndex`, `SMOGIndex`, `DaleChallIndex`, `complex_words_dc`, `paragraphs`
- Variables linguistiques ajoutées : `basicwords_fr`, `tobeverb_fr`, `auxverb_fr`, `conjunction_fr`, `preposition_fr`, `pronoun_fr`, `subordination_fr`, `article_fr`, `interrogative_fr`, `nominalization_fr`  
- Comptage des syllabes basé sur `pyphen` (hyphénation FR) (https://pyphen.org/)
- Résultat : 28 caractéristiques extraites (stockées sous `diff_read`) pour utilisation dans les données d’entrée du modèle : `LIX`, `RIX`, `REL`, `KandelMoles`, `Mesnager`, `characters_per_word`, `syll_per_word`, `words_per_sentence` `sentences_per_paragraph`, `type_token_ratio`, `directspeech_ratio`, `characters`, `syllables`, `words`, `wordtypes`, `sentences`, `long_words`, `complex_words`, `complex_words_mes`, `tobeverb`, `auxverb`, `conjunction`, `preposition`, `nominalization`, `subordination`, `article`, `pronoun`, `interrogative`


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
- 28 caractéristiques linguistiques et de lisibilité (package `readability` adapté au FR) (diff : simple - original)
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
- RandomForestRegressor
  `n_estimators=100`, `random_state=42`
- XGBRegressor
  `n_estimators=100`, `learning_rate=0.1`, `random_state=42`


### Expérimentations :

Exp. 1 :

Données d’entrée : 

(diff_emb_camem_mean_pooling) + (diff_read)

| Modèle         | MAE   | RMSE  | Pearson | Spearman |
|----------------|-------|-------|---------|----------|
| MLP            | 0.642 | 0.827 | 0.372   | 0.346    |
| Random Forest  | 0.532 | 0.728 | 0.318   | 0.378    |
| XGBoost        | 0.562 | 0.708 | 0.416   | 0.440    |



Exp. 2 : 

Données d’entrée : 

(diff_emb_camem_mean_pooling) + (diff_read) + (diff_coref)

| Modèle         | MAE   | RMSE  | Pearson | Spearman |
|----------------|-------|-------|---------|----------|
| MLP            | 0.564 | 0.726 | 0.459   | 0.473    |
| Random Forest  | 0.531 | 0.731 | 0.310   | 0.383    |
| XGBoost        | 0.562 | 0.708 | 0.416   | 0.440    |



Exp. 3 :

Données d’entrée : 

(diff_emb_camem_mean_pooling_pca) + (diff_read) + (diff_coref)

| Modèle         | MAE   | RMSE  | Pearson | Spearman |
|----------------|-------|-------|---------|----------|
| MLP            | 0.520 | 0.668 | 0.490   | 0.510    |
| Random Forest  | 0.508 | 0.689 | 0.440   | 0.498    |
| XGBoost        | 0.571 | 0.767 | 0.228   | 0.276    |


Exp.4 :

Données d’entrée : 

(diff_emb_camem_att_pooling_pca) + (diff_read) + (diff_coref)

| Modèle         | MAE   | RMSE  | Pearson | Spearman |
|----------------|-------|-------|---------|----------|
| MLP            | 0.522 | 0.669 | 0.487   | 0.502    |
| Random Forest  | 0.484 | 0.674 | 0.497   | 0.551    |
| XGBoost        | 0.555 | 0.770 | 0.239   | 0.341    |


Exp.5 : (Meilleurs résultats actuels) ********

Données d’entrée : 

(diff_emb_camem_max_pooling_pca) + (diff_read) + (diff_coref)

| Modèle         | MAE   | RMSE  | Pearson | Spearman |
|----------------|-------|-------|---------|----------|
| MLP            | 0.524 | 0.669 | 0.488   | 0.502    |
| Random Forest  | 0.475 | 0.651 | 0.558   | 0.616    |
| XGBoost        | 0.496 | 0.686 | 0.442   | 0.513    |

Exp. 6 :

— Approche ensemble *(en cours)*


