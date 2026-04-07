### Mémo : Introduction à la méthodologie Machine Learning (ML)

**Ressources de référence :**
* **MOOC scikit-learn (skl) :** [https://inria.github.io/scikit-learn-mooc/](https://inria.github.io/scikit-learn-mooc/)
* **Thèse d’Olivier :** [https://theses.hal.science/tel-04877060](https://theses.hal.science/tel-04877060)

**Nomenclature :** **X** pour les entrées (*inputs*), **Y** pour les sorties (*outputs*).

---

### Méthodologie classique en ML

**1. Analyse des dépendances**
* Analyser les corrélations entre les données. 
* Identifier et supprimer les **X** sans influence manifeste sur les **Y**. 
> Les métriques linéaires (Pearson) peuvent être complétées par une analyse de l'importance des variables pour capter les effets non-linéaires.

**2. Analyse des distributions et filtrage**
* Tracer l'histogramme de chaque **Y**. 
* Identifier les valeurs aberrantes (*outliers*). Un écart supérieur à $3\sigma$ (ou $5\sigma$ pour les royalistes) est généralement le signe d'un problème numérique.
* Vérifier la légitimité physique des points : souvent, les outliers ne sont pas liés à la physique (non-linéarité), mais à un échec du code (non-convergence). Dans ce cas, les filtrer systématiquement.

**3. Normalisation des données**
* Normaliser les données (dans skl via `StandardScaler` ou `MinMaxScaler`). 
* Cette étape est indispensable pour la majorité des techniques (SVD, Régressions régularisées, Réseaux de neurones), à l'exception des méthodes basées sur les forêts aléatoires (random forest).

**4. Séparation Apprentissage / Test (Règle d'or)**
* Scinder la base de données (ex. : 80% pour l'entraînement, 20% pour le test). 
* Ne jamais évaluer un modèle sur les données ayant servi à son entraînement.
> ⚠️ Utiliser exclusivement les coefficients de normalisation calculés sur le jeu d'entraînement pour transformer le jeu de test (éviter le *Data Leakage*).



**5. Réduction de dimension (SVD)**
* Réaliser une SVD (ou POD) pour capter la dimension réelle du problème. 
* Comparer cette dimension au nombre de fonctions utilisées dans des méthodes de type EIM. Si la dimension est faible (20 à 30 valeurs propres), privilégier des méthodes simples. Le nombre de monômes d'une méthode polynomiale sera alors proche du nombre de valeurs propres.

**6. Régression polynomiale régularisée**
* Dans skl utiliser un `Pipeline` : `PolynomialFeatures` suivi d'un modèle linéaire régularisé (`Ridge` ou `Lasso`).
* Ajuster précisément le paramètre de régularisation ($\alpha$). Il reflète le niveau de bruit des données.
* Augmenter la régularisation jusqu'à observer un impact sur le $R^2$ (évalué sur le test). 
> *Disclaimer:* les codes déterministes sont bruités (ex. : précision de convergence d'une nappe de puissance entre 0,1% et 0,01%). Le modèle de ML ne peut pas être plus précis que ce plancher de bruit.

**7. Analyse de l'histogramme des erreurs**
* Analyser le résidu (delta entre **Y** réel et **Y** prédit). 
* Identifier si les erreurs restantes sont liées à des non-linéarités physiques complexes ou à des aberrations statistiques (points incohérents produits par un dysfonctionnement du code).

> Le chapitre 2 de la thèse d’Olivier constitue une excellente mise en œuvre pratique de ces étapes.

---

### Le point critique : La gestion des Outliers

L'analyse des outliers est l'étape la plus cruciale. Le physicien a tendance à surestimer la qualité de ses simulations, alors que les sorties des codes sont fréquemment parasitées par des défauts de convergence (ex. : recherche de barre critique).

La gestion et le filtrage des points aberrants priment sur le choix de la méthode de ML. À données propres, la plupart des méthodes correctement maîtrisées convergent vers des résultats similaires. À l'inverse, les outliers faussent les métriques d'erreur et poussent souvent à choisir des méthodes complexes qui font de l'overfitting (ex. : deep RN) là où une méthode régularisante (ex. : polynômes de bas ordre) capterait mieux la physique.

---

### Vers des méthodes avancées

Une fois l'étape 7 validée, il est possible d'explorer des approches plus sophistiquées :

* **Méthodes scikit-learn avancées :** Random Forest, méthodes polynomiales bayésiennes, Processus Gaussiens (GP).
* **LazyLMC (MOGP) :** Tester cette méthode (développée par Olivier) pour exploiter les corrélations entre les différents **Y**. Elle permet de produire un modèle très compact sans phase d'entraînement lourde. Grace au LOO (Leave One Out) la base de test n'est pas obligatoire.
* **Réseaux de Neurones (RN) :** Utiliser les versions basiques de scikit-learn ou passer sur PyTorch/TensorFlow pour du Deep Learning. Bien que puissants, ils exigent généralement un volume de données et un temps d'apprentissage supérieurs, sauf en cas d'utilisation d'architectures optimisées.
