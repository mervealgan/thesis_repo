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
ENQUÊTE
----------------------------------------------------

----------------------------------------------------
MODÈLE
----------------------------------------------------

### Objectif :
- Prédire le gain de lisibilité perçu entre une phrase originale et sa version simplifiée.


### Options de données d’entrée :
**diff_read**  
- 29 caractéristiques linguistiques et de lisibilité (extraites avec la version FR du package `readability`)  
- Exemple :  
  `diff_LIX`, `diff_RIX`, `diff_REL`, `diff_KandelMoles`, `diff_Mesnager`, `diff_long_words`, `diff_complex_words`, `diff_nominalization`, `diff_subordination`, etc.

**diff_emb_camem**
- 768 dimensions : `embed_diff_0` à `embed_diff_767`  
- Différences entre les vecteurs moyens (mean pooling) des phrases originale et simplifiée, extraits avec CamemBERT (`embedding_simplifiée - embedding_originale`)

**diff_emb_camem_pca**
- 250 composantes principales : version réduite de `diff_emb_camem` par PCA.  
- Utilisée pour améliorer la stabilité du modèle et réduire la dimensionnalité sans perte significative d'information.

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

Données d’entrée : (diff_emb_camem) + (diff_read)

| Modèle         | MAE   | RMSE  | Pearson | Spearman |
|----------------|-------|-------|---------|----------|
| MLP            | 0.642 | 0.827 | 0.372   | 0.346    |
| Random Forest  | 0.532 | 0.728 | 0.318   | 0.378    |
| XGBoost        | 0.562 | 0.708 | 0.416   | 0.440    |



Exp. 2 : 

Données d’entrée : (diff_emb_camem) + (diff_read) + (diff_coref)

| Modèle         | MAE   | RMSE  | Pearson | Spearman |
|----------------|-------|-------|---------|----------|
| MLP            | 0.564 | 0.726 | 0.459   | 0.473    |
| Random Forest  | 0.531 | 0.731 | 0.310   | 0.383    |
| XGBoost        | 0.562 | 0.708 | 0.416   | 0.440    |



Exp. 3 :

Données d’entrée : (diff_emb_camem_pca) + (diff_read) + (diff_coref)


| Modèle         | MAE   | RMSE  | Pearson | Spearman |
|----------------|-------|-------|---------|----------|
| MLP            | 0.520 | 0.668 | 0.490   | 0.510    |
| Random Forest  | 0.508 | 0.689 | 0.440   | 0.498    |
| XGBoost        | 0.571 | 0.767 | 0.228   | 0.276    |



