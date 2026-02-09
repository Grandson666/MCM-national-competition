import numpy as np
import pandas as pd


# Step 1: Load all data files
print("Loading data...")

# Read raw data, fan votes data and total votes data
raw_df = pd.read_csv('datasets/2026_MCM_Problem_C_Data.csv')
fan_votes_df = pd.read_csv('datasets/Fan_Votes.csv')
total_votes_df = pd.read_csv('datasets/Total_Votes.csv')


# Step 2: Feature Engineering
print("Fearure Engineering: processing input features...")

# Basic personal features
feat_df = raw_df[['celebrity_name', 'ballroom_partner', 'celebrity_industry', 'celebrity_homecountry/region', 'celebrity_age_during_season', 'season', 'placement']].copy()

# Construct features (Age and Is_International)
feat_df.rename(columns={'celebrity_age_during_season': 'Age'}, inplace=True)
feat_df['Is_International'] = feat_df['celebrity_homecountry/region'].apply(lambda x: 0 if str(x).strip() == 'United States' else 1)

# Industry mapping
industry_map = {
    'Actor/Actress': 'Performing_Arts', 'Producer': 'Performing_Arts',
    'Athlete': 'Athletes', 'Racing Driver': 'Athletes',
    'Fitness Instructor': 'Athletes', 'Military': 'Athletes',
    'Singer/Rapper': 'Vocal_Artists', 'Musician': 'Vocal_Artists',
    'TV Personality': 'Media_Fashion', 'Social Media Personality': 'Media_Fashion',
    'Model': 'Media_Fashion', 'Beauty Pagent': 'Media_Fashion',
    'Social media personality': 'Media_Fashion', 'Entrepreneur': 'Professional',
    'News Anchor': 'Professional', 'Journalist': 'Professional',
    'Astronaut': 'Professional', 'Politician': 'Professional',
    'Con artist': 'Professional', 'Conservationist': 'Professional',
    'Radio Personality': 'Professional', 'Motivational Speaker': 'Professional',
    'Fashion Designer': 'Professional', 'Sports Broadcaster': 'Professional'
}
feat_df['Industry_Group'] = feat_df['celebrity_industry'].map(industry_map).fillna('Professional')

# Calculate pro features
industry_benchmarks = feat_df.groupby('Industry_Group')['placement'].mean().to_dict()

feat_df['Pro_Experience'] = 0
feat_df['Pro_Avg_Placement'] = 0.0
feat_df['Pro_Pedigree'] = 0.0
feat_df['Pro_Lift'] = 0.0

global_avg_placement = feat_df['placement'].mean()
seasons = sorted(feat_df['season'].unique())

for s in seasons:
    current_season_mask = (feat_df['season'] == s)
    past_data = feat_df[feat_df['season'] < s]

    # Season 1
    if past_data.empty:
        feat_df.loc[current_season_mask, 'Pro_Avg_Placement'] = global_avg_placement
        continue

    pro_list = feat_df.loc[current_season_mask, 'ballroom_partner'].unique()
    for pro in pro_list:
        pro_past = past_data[past_data['ballroom_partner'] == pro]
        pro_mask = (feat_df['season'] == s) & (feat_df['ballroom_partner'] == pro)

        if not pro_past.empty:
            feat_df.loc[pro_mask, 'Pro_Experience'] = len(pro_past)
            feat_df.loc[pro_mask, 'Pro_Avg_Placement'] = pro_past['placement'].mean()

            top3_count = len(pro_past[pro_past['placement'] <= 3])
            feat_df.loc[pro_mask, 'Pro_Pedigree'] = top3_count / len(pro_past)

            lift_vals = pro_past['Industry_Group'].map(industry_benchmarks) - pro_past['placement']
            feat_df.loc[pro_mask, 'Pro_Lift'] = lift_vals.mean()

        else:
            feat_df.loc[pro_mask, 'Pro_Avg_Placement'] = global_avg_placement

# Industry one-hot encoding
industry_dummies = pd.get_dummies(feat_df['Industry_Group'], prefix='Ind')
industry_dummies = industry_dummies.astype(int)
feat_df = pd.concat([feat_df, industry_dummies], axis=1)

# Construct input features
feature_columns_to_keep = ['celebrity_name', 'season', 'Age', 'Is_International', 'Pro_Experience', 'Pro_Avg_Placement', 'Pro_Pedigree', 'Pro_Lift'] + list(industry_dummies.columns)
df_features_clean = feat_df[feature_columns_to_keep].copy()


# Step 3: Build target variable 1 (Judge Scores)
print("\nBuilding Judge Scores target variable...")

judge_scores_records = []

for idx, row in raw_df.iterrows():
    name = row['celebrity_name']
    season = row['season']

    # max 11 weeks and 4 judges per season
    for w in range(1, 12):
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)] 
        existing_cols = [c for c in cols if c in raw_df.columns]
        current_scores = []

        for c in existing_cols:
            val = row[c]

            if pd.isna(val) or str(val).strip().upper() == 'N/A' or val == 0:
                continue
            try:
                current_scores.append(float(val))
            except:
                continue

        if len(current_scores) > 0:
            total_score = sum(current_scores)
            judge_scores_records.append({
                'celebrity_name': name,
                'season': season,
                'week': w,
                'Target_Judge_Scores_Sum': total_score
            })

df_weekly_scores = pd.DataFrame(judge_scores_records)


# Step 4: Build target variable 2 (Fan Votes)
print("Building Fan Votes target variable...")

fan_votes_df.columns = [c.strip().lower() for c in fan_votes_df.columns]
total_votes_df.columns = [c.strip().lower() for c in total_votes_df.columns]

total_votes_df['total_votes_count'] = total_votes_df['predicted_votes_million'] * 1e6
df_fan_share_subset = fan_votes_df[['season', 'week', 'name', 'estimated_fan_share']].copy()

df_fan_votes_calc = pd.merge(
    df_fan_share_subset,
    total_votes_df,
    on=['season', 'week'],
    how='inner'
)

df_fan_votes_calc['Target_Fan_Votes'] = df_fan_votes_calc['total_votes_count'] * df_fan_votes_calc['estimated_fan_share']
df_fan_votes_calc.rename(columns={'name': 'celebrity_name'}, inplace=True)

# Name cleaning
feature_name_clean = lambda x: str(x).strip()
df_features_clean['celebrity_name'] = df_features_clean['celebrity_name'].apply(feature_name_clean)
df_weekly_scores['celebrity_name'] = df_weekly_scores['celebrity_name'].apply(feature_name_clean)
df_fan_votes_calc['celebrity_name'] = df_fan_votes_calc['celebrity_name'].apply(feature_name_clean)


# Step 5: Generate final dataset
print("Merging all data...")

# Merge target features (Judge Scores & Fan Votes)
df_targets = pd.merge(
    df_weekly_scores,
    df_fan_votes_calc[['season', 'week', 'celebrity_name', 'Target_Fan_Votes']],
    on=['season', 'week', 'celebrity_name'],
    how='inner'
)

# Merge input features
final_dataset = pd.merge(
    df_targets,
    df_features_clean,
    on=['season', 'celebrity_name'],
    how='left'
)


# Step 6: Data cleaning and saving
# Remove null values
initial_len = len(final_dataset)
final_dataset.dropna(subset=['Target_Judge_Scores_Sum', 'Target_Fan_Votes'], inplace=True)
print(f"Kept {len(final_dataset)} / {initial_len} samples.")

final_dataset.sort_values(by=['season', 'week', 'celebrity_name'], inplace=True)

# Determine column order
cols_order= [
    'season', 'week', 'celebrity_name',
    'Target_Judge_Scores_Sum', 'Target_Fan_Votes',
    'Age', 'Is_International',
    'Pro_Experience', 'Pro_Avg_Placement', 'Pro_Pedigree', 'Pro_Lift'
]
ind_cols = [c for c in final_dataset.columns if c.startswith('Ind_')]
cols_order += ind_cols

# Determine final dataset
final_dataset = final_dataset[cols_order]

# Save final dataset
output_filename = 'datasets/DWTS_XGBoost_Training_Data.csv'
final_dataset.to_csv(output_filename, index=False)
print(f"\nXGBoost training data successfully saved as: {output_filename}")
