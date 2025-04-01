import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.streamlit_utils import plotly_map, fixNaN, get_lieux_compteurs_df

@st.cache_data
def load_raw_data():
    df_2023 = pd.read_csv('data/raw/2023-comptage-velo-donnees-compteurs.csv', sep=';')
    df_2024 = pd.read_csv('data/raw/2024-comptage-velo-donnees-compteurs.csv', sep=';') 
    df = pd.concat([df_2023, df_2024], axis=0)
    df = fixNaN(df)
    return df

# Charger les données
df = pd.read_csv("data/processed/lieu-compteur-one-hot-encoded.csv", index_col=0)

st.image("streamlit_assets/banniere6.jpeg", use_container_width=True)

#Titres
st.markdown("""
# **Projet Data Science - Trafic Cycliste à Paris**  
### _de Janvier à Mars 2025_
""")
st.sidebar.title("Sommaire")
pages=["Présentation du Projet", "Dataset", "Pre-processing", "Visualisation des données", "Modèles de classification", "Modèles de régression", "Interprétation et résultats", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)


if page == pages[0]:
    st.markdown("## I. Contexte")

    st.markdown("""
Face à l'essor du vélo comme mode de transport durable, la Ville de Paris a mis en place, depuis plusieurs années, un réseau de compteurs à vélo permanents pour mesurer l'évolution de la pratique cycliste dans la capitale.

Ces capteurs, installés sur les axes clés de la ville, collectent en continu des données sur le flux des cyclistes.
    """)

    st.image("streamlit_assets/comptagevélo.jpeg", use_container_width=True)

    st.markdown("""
Ce projet s'inscrit dans une démarche de transition vers une mobilité plus verte et une volonté d'adapter les infrastructures urbaines aux besoins réels, tel que proposé dans le plan vélo 2021-2026 d'aménagement de pistes cyclables de la mairie de Paris.

L'enjeu est de transformer ces données brutes en insights exploitables, permettant d'éclairer les décisions publiques de manière objective et data-driven.
    """)





st.markdown("---")


if page == pages[0]: 
    st.markdown("## II. Objectifs")

    st.markdown("""
Ce projet vise à développer un **outil prédictif du trafic cycliste à Paris**, en exploitant les données historiques des compteurs vélo.

### Objectifs principaux :
- Identifier les **tendances d’usage** (heures de pointe, zones saturées, variations saisonnières).
- Générer des **visualisations claires** (cartes thermiques, graphiques temporels).
- Aider à la **prise de décision** sur les aménagements à prioriser.

### Bénéfices pour la Mairie de Paris :
- Prioriser les **aménagements ciblés** (pistes élargies, carrefours sécurisés, nouveaux itinéraires).
- Évaluer l’**impact des politiques existantes**.
- **Optimiser le réseau cyclable** à long terme.

### Ambition finale :
> Réduire les **conflits d’usage**, améliorer la **sécurité**, et encourager la pratique du vélo grâce à une **planification data-driven**, combinant **rétrospective** et **prédiction** pour une mobilité plus fluide et résiliente.
    """)









if page == pages[1]: 

    st.markdown("""
### 1. Source des données""")  
                
    st.image("streamlit_assets/opendata2.png", use_container_width=True)

    st.markdown("""
- Utilisation des jeux de données ouverts proposés par la Ville de Paris via [opendata.paris.fr](https://opendata.paris.fr) :

  - le jeu de données [Comptage vélo - Données compteurs](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information) pour les données de 2024.
  - le jeu de données [Comptage vélo - Historique - Données Compteurs et Sites de comptage](https://opendata.paris.fr/explore/dataset/comptage-velo-historique-donnees-compteurs/information) pour les données de 2023.
                 
- Les données sont publiées sous la licence Open Database License (ODbL), qui autorise la réutilisation, l’adaptation et la création de travaux dérivés à partir de ces jeux de données, à condition d’en citer la source.

---
                
### 2. Période 
                
Les données sont mises à jour quotidiennement. 

Nous avons récupéré toutes les données du 1er janvier 2023 au 28 février 2025 (26 mois).                

---

### 3. Contenu des jeux de données  
Les jeux de données recensent les comptages horaires de vélos effectués par environ une centaine de compteurs répartis dans Paris.
                  
Chaque lieu de comptage est généralement équipé de deux compteurs, positionnés pour mesurer le trafic dans chaque direction d’une même rue. 
                 
Au total, pour la période 2023-2024, le jeu de données contient environ 1,8 million de lignes réparties sur 16 variables.

---

### 4. Structure des données  
Chaque ligne du dataset correspond au nombre de vélos enregistrés pendant une heure par un compteur donné.  
Les données incluent, en plus du comptage horaire, plusieurs métadonnées associées au compteur ou au site de comptage, telles que :
- Le nom et l’identifiant du compteur  
- Le lieu de comptage  
- La date d’installation  
- Les coordonnées géographiques  
- Éventuellement, des liens vers des photos du site

---

### 5. Nettoyage et sélection des variables  
Afin de simplifier et d’optimiser l’analyse, nous avons supprimé les variables non pertinentes pour l'entraînement du modèle, en particulier les champs techniques ou visuels comme les liens vers les photos, les identifiants internes ou le type d’image.  

Voici un extrait de notre dataset avec les variables que nous avons décidé de conserver :""")

    st.dataframe(load_raw_data().sample(5))
    # st.image("streamlit_assets/dataframe.jpeg", use_container_width=True)

    st.markdown("""
---

### 6. Objectif d’analyse et variable cible  
L’objectif de notre étude est de prédire le nombre de vélos comptés pendant une heure sur un compteur donné.  
La variable cible de notre modèle est donc le comptage horaire, un indicateur clé pour analyser l'évolution de la circulation cycliste dans Paris.

---

### 7. Forces et limites du dataset  
Le dataset se distingue par sa précision horaire et sa couverture géographique dense, ce qui permet d’identifier des tendances temporelles comme les variations quotidiennes ou saisonnières du trafic cycliste.  
Cependant, il ne contient pas d’informations contextuelles telles que :
- La météo  
- La présence d’événements particuliers (manifestations, grèves, festivals)  
- Ou des données sociodémographiques comme la densité de population par zone  

Cette absence limite la profondeur des analyses prédictives que l’on peut mener.
""")

if page == pages[2]: 

    st.markdown("""
    ### 1. Suppression des NaN 
        
    Certaines variables de métadonnées des compteurs ("Identifiant du compteur", "Coordonnées géographiques", ...) ont des valeurs NaN (environ 3.4% sur le dataset)

    Plusieurs compteurs du dataset correspondaient en réalité à un même emplacement, ce qui a permis de réduire les NaN en les renommant et fusionnant. 

    Les derniers NaN provenaient de deux compteurs atypiques, finalement supprimés pour obtenir un dataset complet et sans valeurs manquantes

        
    ### 2. Conversion Date au format datetime
                    
    Variable "Date et heure de comptage" convertie au format datetime de 
    Pandas. (fuseau horaire Europe/Paris afin de correctement capturer 
    les tendances journalières)
    """) 
                    
    st.code("""
    # Convertir la colonne en datetime (avec gestion du fuseau horaire)
    df["Date et heure de comptage"] = pd.to_datetime(df["Date et heure de comptage"], utc=True)
    df["Date et heure de comptage"] = df["Date et heure de comptage"].dt.tz_localize(None)
    """, language="python")

    st.markdown("""                
    ### 3. Ajout de variables
                    
    La variable "Date et heure de comptage" décomposée en variables "année", 
    "mois", "jour", "jour de la semaine" et "heure" afin de faciliter la data visualisation et voir si 
    certaines de ces variables étaient corrélés à notre variable cible. """)
    
    st.code("""
    df["Jour"] = df["Date et heure de comptage"].dt.date
    df["Mois"] = df["Date et heure de comptage"].dt.month
    df["Année"] = df["Date et heure de comptage"].dt.year
    df["Heure"] = df["Date et heure de comptage"].dt.hour
    """, language="python")

    st.markdown("""   
    
    Ajout des variables catégorielles binaires "Week-end", "Jour fériés" et "Vacances scolaires" afin de mesurer si les jours non travaillés ont un impact sur la pratique cyclable.  

    ### 4. Normalisation des données

    Nous avons appliqué deux types de **normalisation** sur les colonnes temporelles et contextuelles, notamment pour réduire l'impact des valeurs extrêmes de Comptage horaire sur les prédictions de notre modèle,  la variable Comptage horaire ne suivant pas une loi normale :

    1. **Standardisation** : centre les données autour de 0 avec une variance de 1.
    2. **Min-Max Scaling** : transforme les valeurs dans une plage définie, ici entre 0 et 1.
        """)

    st.markdown("### 🔹 Standardisation")

    st.code("""
    from sklearn.preprocessing import StandardScaler

col_norm = ["Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]

scaler = StandardScaler()
df[col_norm] = scaler.fit_transform(df[col_norm])
    """, language="python")

    st.markdown("### 🔹 Normalisation Min-Max")

    st.code("""
from sklearn.preprocessing import MinMaxScaler

col_norm = ["Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]

scaler = MinMaxScaler(feature_range=(0, 1))
df[col_norm] = scaler.fit_transform(df[col_norm])
    """, language="python")

    st.markdown("""
    Ces transformations permettent de préparer les données pour les modèles sensibles à l’échelle des variables (régressions, KNN, etc.).
    """)

    

    st.write("### Extrait du Dataframe après pré-processing")
    st.dataframe(df.head(10))




                
if page == pages[3]: 
    raw_data = load_raw_data()

    st.write("### Visualisation des données")

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    ### I. Cartographie

    Carte de la ville de Paris représentant les positions des différents compteurs du dataset (La taille de chaque point correspond au comptage horaire total).""")

    st.plotly_chart(plotly_map(load_raw_data()))

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    Compteurs sont répartis sur les axes principaux :
            
    - Sud-Ouest / Nord-Est (avenue Denfert-Rochereau et boulevard Sébastopol) 
    - Est-Ouest (avenue de la Grande Armée et avenue des Champs-Élysées). 
    - Les quais de la Seine ainsi que le boulevard périphérique Sud (le long de la voie de tram T3a) sont aussi couverts. 
                
    Boulevard périphérique nord et les 17 et 18e arrondissements n'ont pas de compteurs. 
    
    Compteurs "centraux" ont plus de comptage que ceux en périphérie de Paris : 
    Corrélation entre la localisation du compteur et le comptage horaire ?


    ### II. Évolution temporelle""")

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown(""" 
    ##### a. Saisonnalité du trafic""")
    
    fig = plt.figure()
    sns.barplot(df, x='Mois', y='Comptage horaire', errorbar=None)
    plt.xlabel("Mois")
    plt.xticks(rotation=45)
    plt.title("Comptage horaire moyen en fonction du mois");
    st.pyplot(fig)

    st.markdown("""
    <div style='font-size:18px;'>
    On constate une baisse du comptage en <strong>hiver</strong> (janvier et décembre) et en <strong>été</strong> (au mois d'août).<br>
    Cela est peut-être dû aux <strong>vacances</strong>, à certains <strong>événements</strong> (JO de Paris en août) et à la <strong>météo</strong> (il fait plus froid en hiver, ce qui n'encourage pas la pratique cycliste).
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    ##### b. Comportement selon les jours""")
    
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(df, x='Jour_semaine', y='Comptage horaire', errorbar=None)
    plt.xlabel("Jour du mois")
    plt.title("Comptage horaire moyen en fonction du jour de la semaine");
    st.pyplot(fig)

    st.markdown("""
    <div style='font-size:18px;'>
    On constate également plus de <strong>comptages</strong> du <strong>lundi au vendredi</strong>, ce qui correspond aux <strong>trajets domicile-travail</strong>.<br>
    En moyenne, il y a environ <strong>50% de <strong>vélos</strong> en plus</strong> en <strong>semaine</strong> par rapport au <strong>week-end</strong>.
    </div>
    """, unsafe_allow_html=True)


    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    ##### c. Evolution du trafic au fil des heures""")

    st.markdown("""
    <div style='font-size:18px;'>
    À gauche : <strong>jours de la semaine</strong> (lundi à vendredi) — À droite : <strong>week-end</strong> (samedi et dimanche)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # Filtrage
    df_semaine = df[df['Jour_semaine'].isin([1, 2, 3, 4, 5])]
    df_weekend = df[df['Jour_semaine'].isin([6, 7])]

    # Détermination du même axe Y pour les deux graphiques
    y_max = max(df['Comptage horaire'].max(), 1)  # max global pour fixer l'échelle

    # Création des deux graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    # Graphique semaine
    sns.barplot(ax=axes[0], data=df_semaine, x='Heure', y='Comptage horaire', errorbar=None)
    axes[0].set_title("Comptage horaire moyen (lundi à vendredi)")
    axes[0].set_xlabel("Heure de la journée")
    axes[0].set_ylabel("Comptage horaire")
    axes[0].set_ylim(0, 400)

    # Graphique week-end
    sns.barplot(ax=axes[1], data=df_weekend, x='Heure', y='Comptage horaire', errorbar=None)
    axes[1].set_title("Comptage horaire moyen (week-end)")
    axes[1].set_xlabel("Heure de la journée")
    axes[1].set_ylabel("")  # on peut laisser vide pour alléger visuellement
    axes[1].set_ylim(0, 400)

    st.pyplot(fig)

    st.markdown("""
    <div style='font-size:18px;'>
    <strong>Forte augmentation du trafic</strong> aux <strong>heures de pointe</strong> (8h–9h / 18h–19h) en semaine.<br>
    <strong>Volume de passages</strong> relativement <strong>régulier</strong> entre <strong>11h et 20h</strong> le <strong>week-end</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### III. Distribution de la variable cible""")

    st.markdown('#### Boxplot de la variable comptage horaire')

    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(raw_data, x='Comptage horaire')
    st.pyplot(fig)

    st.markdown('#### Boxplot de la variable comptage horaire en fonction du lieu de comptage')

    agg_data = get_lieux_compteurs_df(raw_data)
    site = st.selectbox("Nom du site de comptage", list(agg_data['Nom du site de comptage'].unique()))
    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(agg_data.loc[agg_data['Nom du site de comptage'] == site], x='Comptage horaire')
    plt.title(site)
    st.pyplot(fig)
    
    st.markdown("""
    ### IV. Corrélation entre les variables""")
    
    st.image("streamlit_assets/matrice.jpeg", use_container_width=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <strong>Matrice de corrélation entre les variables</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <ul>
    <li>Le <strong>comptage horaire</strong> est légèrement corrélé au <strong>nom du compteur</strong> et à <strong>l'heure de la journée</strong>.</li>
    <li>Corrélation forte entre les variables <strong>jour_semaine</strong> et <strong>week-end</strong> (variables potentiellement redondantes).</li>
    <li>Forte corrélation entre <strong>"date et heure de comptage"</strong>, <strong>année</strong> et <strong>mois</strong>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

   
