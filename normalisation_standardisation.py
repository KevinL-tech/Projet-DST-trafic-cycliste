import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

df_2023 = pd.read_csv('2023-comptage-velo-donnees-compteurs.csv', sep=';')
df_2024 = pd.read_csv('2024-comptage-velo-donnees-compteurs.csv', sep=';')
df = pd.concat([df_2023, df_2024])

df = df.drop(index=df.loc[df['Nom du compteur'] == '10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike IN]'].index)
df = df.drop(index=df.loc[df['Nom du compteur'] == '10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike OUT]'].index)
df = df.replace({
    'Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO': 'Face au 48 quai de la marne NE-SO',
    'Face au 48 quai de la marne Face au 48 quai de la marne Vélos SO-NE': 'Face au 48 quai de la marne SO-NE',
    'Totem 64 Rue de Rivoli Totem 64 Rue de Rivoli Vélos E-O': 'Totem 64 Rue de Rivoli E-O',
    'Totem 64 Rue de Rivoli Totem 64 Rue de Rivoli Vélos O-E': 'Totem 64 Rue de Rivoli O-E',
    'Quai des Tuileries Quai des Tuileries Vélos NO-SE': 'Quai des Tuileries NO-SE',
    'Quai des Tuileries Quai des Tuileries Vélos SE-NO': 'Quai des Tuileries SE-NO',
    'Pont des Invalides (couloir bus)': 'Pont des Invalides',
    '69 Boulevard Ornano (temporaire)': '69 Boulevard Ornano',
    '30 rue Saint Jacques (temporaire)': '30 rue Saint Jacques',
    '27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE': '27 quai de la Tournelle NO-SE',
    '27 quai de la Tournelle 27 quai de la Tournelle Vélos SE-NO': '27 quai de la Tournelle SE-NO',
    'Pont des Invalides (couloir bus) N-S': 'Pont des Invalides N-S',
})

df['comptage_datetime'] = pd.to_datetime(df['Date et heure de comptage'], errors='coerce', utc=True)
df['comptage_datetime'] = df['comptage_datetime'].dt.tz_convert('Europe/Paris')

df['heure'] = df['comptage_datetime'].dt.hour
df['jour_semaine'] = df['comptage_datetime'].dt.weekday
df['mois'] = df['comptage_datetime'].dt.month

# Ajout variables binaires 
df['weekend'] = df['jour_semaine'].apply(lambda x: 1 if x >= 5 else 0)
# il manque les jours fériés et les vacances scolaires

#  Encodage des variables catégorielles
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['nom_compteur_encoded'] = le.fit_transform(df['Nom du compteur'])

plt.figure(figsize=(10,6))
sns.histplot(df['Comptage horaire'], bins=50, kde=True)
plt.title('Distribution du comptage horaire avant normalisation')
plt.show()

X = df[['heure', 'jour_semaine', 'mois', 'weekend', 'nom_compteur_encoded']]
y = df['Comptage horaire']

# Matrice de corrélation
plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation entre les caractéristiques')
plt.show()

scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
sns.boxplot(data=X, ax=axes[0]).set_title('Avant normalisation')
sns.boxplot(data=X_minmax, ax=axes[1]).set_title('Après normalisation MinMax')
plt.show()


scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

plt.figure(figsize=(10,6))
sns.histplot(X_std[:,0], kde=True) 
plt.title('Distribution de l\'heure après standardisation')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

