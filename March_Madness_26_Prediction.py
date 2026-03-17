# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:39:34 2026

@author: lotrc
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
# ─────────────────────────────────────────────────────────────────────
# 1. LOAD & TRAIN MODEL (same as 2025 pipeline)
# ─────────────────────────────────────────────────────────────────────
df = pd.read_excel('/mnt/user-data/uploads/ncaa_tournament_dataset.xlsx', sheet_name='Tournament Dataset')

ROUND_ORDER = {
    'First Round': 1, 'Round of 32': 2, 'Sweet 16': 3,
    'Elite Eight': 4, 'Final Four': 5, 'Champion': 6, '—': 0
}
df['Round_Num'] = df['Round Reached'].map(ROUND_ORDER)

def make_matchup_features(t1_stats, t2_stats):
    feats = {}
    cols = ['Net Rtg (AdjEM)', 'Adj ORtg', 'Adj DRtg', 'Adj Tempo', 'Luck',
            'Pomeroy Rank', 'Reg Wins', 'Reg Losses', 'Seed']
    for c in cols:
        v1 = t1_stats.get(c, 0) or 0
        v2 = t2_stats.get(c, 0) or 0
        try: v1, v2 = float(v1), float(v2)
        except: v1, v2 = 0, 0
        safe_c = c.replace(' ', '_').replace('(', '').replace(')', '')
        feats[f't1_{safe_c}'] = v1
        feats[f't2_{safe_c}'] = v2
        feats[f'diff_{safe_c}'] = v1 - v2
    s1 = float(t1_stats.get('Seed', 8) or 8)
    s2 = float(t2_stats.get('Seed', 8) or 8)
    feats['seed_diff'] = s1 - s2
    feats['higher_seed_is_t1'] = 1 if s1 < s2 else 0
    w1 = float(t1_stats.get('Reg Wins', 0) or 0)
    l1 = float(t1_stats.get('Reg Losses', 1) or 1)
    w2 = float(t2_stats.get('Reg Wins', 0) or 0)
    l2 = float(t2_stats.get('Reg Losses', 1) or 1)
    feats['t1_win_pct'] = w1 / (w1 + l1 + 1e-9)
    feats['t2_win_pct'] = w2 / (w2 + l2 + 1e-9)
    feats['diff_win_pct'] = feats['t1_win_pct'] - feats['t2_win_pct']
    return feats

all_matchup_records = []
for year in df['Year'].unique():
    ydf = df[df['Year'] == year].copy()
    ydf = ydf.dropna(subset=['Seed'])
    ydf['Seed'] = ydf['Seed'].astype(int)
    by_seed = {}
    for _, row in ydf.iterrows():
        s = int(row['Seed'])
        if s not in by_seed: by_seed[s] = []
        by_seed[s].append(row)
    teams_dict = {row['Team']: row.to_dict() for _, row in ydf.iterrows()}

    for seed_pair in [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]:
        s1, s2 = seed_pair
        teams_s1 = sorted(by_seed.get(s1, []), key=lambda r: -r['Round_Num'])
        teams_s2 = sorted(by_seed.get(s2, []), key=lambda r: -r['Round_Num'])
        n = min(len(teams_s1), len(teams_s2))
        for i in range(n):
            t1, t2 = teams_s1[i], teams_s2[i]
            r1, r2 = t1['Round_Num'], t2['Round_Num']
            if r1 == r2: continue
            winner = 1 if r1 > r2 else 0
            feats = make_matchup_features(t1.to_dict(), t2.to_dict())
            feats['year'] = year; feats['round'] = 1; feats['winner'] = winner
            all_matchup_records.append(feats)
            feats_f = make_matchup_features(t2.to_dict(), t1.to_dict())
            feats_f['year'] = year; feats_f['round'] = 1; feats_f['winner'] = 1 - winner
            all_matchup_records.append(feats_f)

    for rnd in range(2, 7):
        winners = [t for t, d in teams_dict.items() if d['Round_Num'] >= rnd]
        losers_at_rnd = [t for t, d in teams_dict.items() if d['Round_Num'] == rnd-1]
        for w_team in winners:
            w = teams_dict[w_team]
            w_seed = int(w['Seed'])
            best_opp, best_dist = None, 999
            for l_team in losers_at_rnd:
                l = teams_dict[l_team]
                dist = abs(w_seed - int(l['Seed']))
                if dist < best_dist: best_dist = dist; best_opp = l_team
            if best_opp and best_dist <= 8:
                l = teams_dict[best_opp]
                feats = make_matchup_features(w, l)
                feats['year'] = year; feats['round'] = rnd; feats['winner'] = 1
                all_matchup_records.append(feats)
                feats_f = make_matchup_features(l, w)
                feats_f['year'] = year; feats_f['round'] = rnd; feats_f['winner'] = 0
                all_matchup_records.append(feats_f)

matchup_df = pd.DataFrame(all_matchup_records)
feature_cols = [c for c in matchup_df.columns if c not in ['year','winner']]
X = matchup_df[feature_cols].fillna(0)
y = matchup_df['winner']

gbm = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, min_samples_leaf=5, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
ensemble = VotingClassifier([('gbm', gbm), ('rf', rf)], voting='soft')
ensemble.fit(X, y)
print("Model trained.")

# ─────────────────────────────────────────────────────────────────────
# 2. 2026 KENPOM DATA
# ─────────────────────────────────────────────────────────────────────
kenpom_2026 = {
    'Duke':         {'Pom_Rank':1,  'Reg_Wins':32, 'Reg_Losses':2,  'NetRtg':38.90, 'ORtg':128.0, 'DRtg':89.1,  'Tempo':65.3,  'Luck':0.049},
    'Arizona':      {'Pom_Rank':2,  'Reg_Wins':32, 'Reg_Losses':2,  'NetRtg':37.66, 'ORtg':127.7, 'DRtg':90.0,  'Tempo':69.8,  'Luck':0.023},
    'Michigan':     {'Pom_Rank':3,  'Reg_Wins':31, 'Reg_Losses':3,  'NetRtg':37.59, 'ORtg':126.6, 'DRtg':89.0,  'Tempo':70.9,  'Luck':0.045},
    'Florida':      {'Pom_Rank':4,  'Reg_Wins':26, 'Reg_Losses':7,  'NetRtg':33.79, 'ORtg':125.5, 'DRtg':91.7,  'Tempo':70.5,  'Luck':-0.036},
    'Houston':      {'Pom_Rank':5,  'Reg_Wins':28, 'Reg_Losses':6,  'NetRtg':33.43, 'ORtg':124.9, 'DRtg':91.4,  'Tempo':63.3,  'Luck':-0.006},
    'Iowa St.':     {'Pom_Rank':6,  'Reg_Wins':27, 'Reg_Losses':7,  'NetRtg':32.42, 'ORtg':123.8, 'DRtg':91.4,  'Tempo':66.5,  'Luck':-0.012},
    'Illinois':     {'Pom_Rank':7,  'Reg_Wins':24, 'Reg_Losses':8,  'NetRtg':32.10, 'ORtg':131.2, 'DRtg':99.1,  'Tempo':65.5,  'Luck':-0.050},
    'Purdue':       {'Pom_Rank':8,  'Reg_Wins':27, 'Reg_Losses':8,  'NetRtg':31.20, 'ORtg':131.6, 'DRtg':100.4, 'Tempo':64.4,  'Luck':-0.006},
    'Michigan St.': {'Pom_Rank':9,  'Reg_Wins':25, 'Reg_Losses':7,  'NetRtg':28.31, 'ORtg':123.0, 'DRtg':94.7,  'Tempo':66.0,  'Luck':0.005},
    'Gonzaga':      {'Pom_Rank':10, 'Reg_Wins':30, 'Reg_Losses':3,  'NetRtg':28.10, 'ORtg':122.0, 'DRtg':93.9,  'Tempo':68.6,  'Luck':0.072},
    'Connecticut':  {'Pom_Rank':11, 'Reg_Wins':29, 'Reg_Losses':5,  'NetRtg':27.87, 'ORtg':122.0, 'DRtg':94.1,  'Tempo':64.4,  'Luck':0.055},
    'Vanderbilt':   {'Pom_Rank':12, 'Reg_Wins':26, 'Reg_Losses':8,  'NetRtg':27.51, 'ORtg':126.8, 'DRtg':99.3,  'Tempo':68.8,  'Luck':0.018},
    'Virginia':     {'Pom_Rank':13, 'Reg_Wins':29, 'Reg_Losses':5,  'NetRtg':26.71, 'ORtg':122.5, 'DRtg':95.8,  'Tempo':65.7,  'Luck':0.056},
    'Nebraska':     {'Pom_Rank':14, 'Reg_Wins':26, 'Reg_Losses':6,  'NetRtg':26.16, 'ORtg':118.5, 'DRtg':92.4,  'Tempo':66.7,  'Luck':0.034},
    'Arkansas':     {'Pom_Rank':15, 'Reg_Wins':26, 'Reg_Losses':8,  'NetRtg':26.05, 'ORtg':127.7, 'DRtg':101.6, 'Tempo':71.0,  'Luck':0.051},
    'Tennessee':    {'Pom_Rank':16, 'Reg_Wins':22, 'Reg_Losses':11, 'NetRtg':26.02, 'ORtg':121.1, 'DRtg':95.0,  'Tempo':65.0,  'Luck':-0.060},
    'St. John\'s':  {'Pom_Rank':17, 'Reg_Wins':28, 'Reg_Losses':6,  'NetRtg':25.91, 'ORtg':120.1, 'DRtg':94.2,  'Tempo':69.6,  'Luck':0.061},
    'Alabama':      {'Pom_Rank':18, 'Reg_Wins':23, 'Reg_Losses':9,  'NetRtg':25.72, 'ORtg':129.0, 'DRtg':103.3, 'Tempo':73.1,  'Luck':0.019},
    'Louisville':   {'Pom_Rank':19, 'Reg_Wins':23, 'Reg_Losses':10, 'NetRtg':25.42, 'ORtg':124.1, 'DRtg':98.6,  'Tempo':69.6,  'Luck':-0.020},
    'Texas Tech':   {'Pom_Rank':20, 'Reg_Wins':22, 'Reg_Losses':10, 'NetRtg':25.22, 'ORtg':125.0, 'DRtg':99.8,  'Tempo':66.2,  'Luck':0.006},
    'Kansas':       {'Pom_Rank':21, 'Reg_Wins':23, 'Reg_Losses':10, 'NetRtg':24.44, 'ORtg':118.3, 'DRtg':93.9,  'Tempo':67.6,  'Luck':0.053},
    'Wisconsin':    {'Pom_Rank':22, 'Reg_Wins':24, 'Reg_Losses':10, 'NetRtg':23.39, 'ORtg':125.3, 'DRtg':102.0, 'Tempo':68.7,  'Luck':0.041},
    'BYU':          {'Pom_Rank':23, 'Reg_Wins':23, 'Reg_Losses':11, 'NetRtg':23.25, 'ORtg':125.5, 'DRtg':102.2, 'Tempo':69.9,  'Luck':-0.017},
    'Saint Mary\'s':{'Pom_Rank':24, 'Reg_Wins':27, 'Reg_Losses':5,  'NetRtg':23.07, 'ORtg':120.3, 'DRtg':97.2,  'Tempo':65.2,  'Luck':0.011},
    'Iowa':         {'Pom_Rank':25, 'Reg_Wins':21, 'Reg_Losses':12, 'NetRtg':22.44, 'ORtg':121.7, 'DRtg':99.3,  'Tempo':63.0,  'Luck':-0.061},
    'Ohio St.':     {'Pom_Rank':26, 'Reg_Wins':21, 'Reg_Losses':12, 'NetRtg':22.24, 'ORtg':124.3, 'DRtg':102.1, 'Tempo':66.1,  'Luck':-0.031},
    'UCLA':         {'Pom_Rank':27, 'Reg_Wins':23, 'Reg_Losses':11, 'NetRtg':21.67, 'ORtg':123.7, 'DRtg':102.1, 'Tempo':64.6,  'Luck':0.017},
    'Kentucky':     {'Pom_Rank':28, 'Reg_Wins':21, 'Reg_Losses':13, 'NetRtg':21.48, 'ORtg':120.5, 'DRtg':99.0,  'Tempo':68.3,  'Luck':-0.019},
    'North Carolina':{'Pom_Rank':29,'Reg_Wins':24, 'Reg_Losses':8,  'NetRtg':20.84, 'ORtg':121.4, 'DRtg':100.5, 'Tempo':67.9,  'Luck':0.057},
    'Utah St.':     {'Pom_Rank':30, 'Reg_Wins':28, 'Reg_Losses':6,  'NetRtg':20.76, 'ORtg':122.1, 'DRtg':101.4, 'Tempo':67.7,  'Luck':0.065},
    'Miami FL':     {'Pom_Rank':31, 'Reg_Wins':25, 'Reg_Losses':8,  'NetRtg':20.68, 'ORtg':121.4, 'DRtg':100.7, 'Tempo':67.6,  'Luck':0.021},
    'Georgia':      {'Pom_Rank':32, 'Reg_Wins':22, 'Reg_Losses':10, 'NetRtg':20.48, 'ORtg':124.7, 'DRtg':104.2, 'Tempo':71.4,  'Luck':-0.005},
    'Villanova':    {'Pom_Rank':33, 'Reg_Wins':24, 'Reg_Losses':8,  'NetRtg':19.97, 'ORtg':120.4, 'DRtg':100.4, 'Tempo':65.2,  'Luck':0.067},
    'N.C. State':   {'Pom_Rank':34, 'Reg_Wins':20, 'Reg_Losses':13, 'NetRtg':19.60, 'ORtg':124.1, 'DRtg':104.5, 'Tempo':69.1,  'Luck':-0.029},
    'Santa Clara':  {'Pom_Rank':35, 'Reg_Wins':26, 'Reg_Losses':8,  'NetRtg':19.40, 'ORtg':123.6, 'DRtg':104.2, 'Tempo':69.2,  'Luck':0.015},
    'Clemson':      {'Pom_Rank':36, 'Reg_Wins':24, 'Reg_Losses':10, 'NetRtg':19.24, 'ORtg':116.5, 'DRtg':97.3,  'Tempo':64.2,  'Luck':0.011},
    'Texas':        {'Pom_Rank':37, 'Reg_Wins':18, 'Reg_Losses':14, 'NetRtg':19.03, 'ORtg':125.0, 'DRtg':105.9, 'Tempo':66.9,  'Luck':-0.083},
    'Auburn':       {'Pom_Rank':38, 'Reg_Wins':17, 'Reg_Losses':16, 'NetRtg':19.02, 'ORtg':124.8, 'DRtg':105.8, 'Tempo':67.2,  'Luck':-0.053},
    'Texas A&M':    {'Pom_Rank':39, 'Reg_Wins':21, 'Reg_Losses':11, 'NetRtg':18.67, 'ORtg':119.7, 'DRtg':101.0, 'Tempo':70.5,  'Luck':-0.002},
    'Oklahoma':     {'Pom_Rank':40, 'Reg_Wins':19, 'Reg_Losses':15, 'NetRtg':18.37, 'ORtg':124.2, 'DRtg':105.8, 'Tempo':66.2,  'Luck':-0.070},
    'Saint Louis':  {'Pom_Rank':41, 'Reg_Wins':28, 'Reg_Losses':5,  'NetRtg':18.32, 'ORtg':119.5, 'DRtg':101.2, 'Tempo':71.0,  'Luck':0.030},
    'SMU':          {'Pom_Rank':42, 'Reg_Wins':20, 'Reg_Losses':13, 'NetRtg':18.09, 'ORtg':122.9, 'DRtg':104.8, 'Tempo':68.5,  'Luck':-0.043},
    'TCU':          {'Pom_Rank':43, 'Reg_Wins':22, 'Reg_Losses':11, 'NetRtg':17.59, 'ORtg':115.4, 'DRtg':97.8,  'Tempo':67.7,  'Luck':0.004},
    'VCU':          {'Pom_Rank':45, 'Reg_Wins':27, 'Reg_Losses':7,  'NetRtg':17.21, 'ORtg':119.9, 'DRtg':102.7, 'Tempo':68.5,  'Luck':-0.007},
    'South Florida':{'Pom_Rank':47, 'Reg_Wins':25, 'Reg_Losses':8,  'NetRtg':16.39, 'ORtg':117.3, 'DRtg':100.9, 'Tempo':71.5,  'Luck':-0.026},
    'Missouri':     {'Pom_Rank':52, 'Reg_Wins':20, 'Reg_Losses':12, 'NetRtg':15.39, 'ORtg':119.5, 'DRtg':104.1, 'Tempo':66.2,  'Luck':0.041},
    'UCF':          {'Pom_Rank':54, 'Reg_Wins':21, 'Reg_Losses':11, 'NetRtg':15.04, 'ORtg':120.5, 'DRtg':105.4, 'Tempo':69.2,  'Luck':0.097},
    'Akron':        {'Pom_Rank':64, 'Reg_Wins':29, 'Reg_Losses':5,  'NetRtg':12.80, 'ORtg':118.8, 'DRtg':106.0, 'Tempo':70.3,  'Luck':0.018},
    'McNeese':      {'Pom_Rank':68, 'Reg_Wins':28, 'Reg_Losses':5,  'NetRtg':12.48, 'ORtg':114.3, 'DRtg':101.8, 'Tempo':66.2,  'Luck':0.084},
    'Northern Iowa':{'Pom_Rank':72, 'Reg_Wins':23, 'Reg_Losses':12, 'NetRtg':11.81, 'ORtg':110.0, 'DRtg':98.2,  'Tempo':62.3,  'Luck':-0.070},
    'Cal Baptist':  {'Pom_Rank':106,'Reg_Wins':25, 'Reg_Losses':8,  'NetRtg':5.99,  'ORtg':107.9, 'DRtg':101.9, 'Tempo':65.8,  'Luck':0.091},
    'High Point':   {'Pom_Rank':92, 'Reg_Wins':30, 'Reg_Losses':4,  'NetRtg':8.40,  'ORtg':117.0, 'DRtg':108.6, 'Tempo':69.9,  'Luck':0.048},
    'Hawaii':       {'Pom_Rank':108,'Reg_Wins':24, 'Reg_Losses':8,  'NetRtg':5.97,  'ORtg':107.1, 'DRtg':101.2, 'Tempo':69.7,  'Luck':0.038},
    'Hofstra':      {'Pom_Rank':87, 'Reg_Wins':24, 'Reg_Losses':10, 'NetRtg':9.49,  'ORtg':114.6, 'DRtg':105.1, 'Tempo':64.7,  'Luck':-0.052},
    'North Dakota St.':{'Pom_Rank':113,'Reg_Wins':27,'Reg_Losses':7,'NetRtg':5.13,  'ORtg':111.7, 'DRtg':106.6, 'Tempo':66.3,  'Luck':0.040},
    'Furman':       {'Pom_Rank':191,'Reg_Wins':22, 'Reg_Losses':12, 'NetRtg':-1.98, 'ORtg':107.5, 'DRtg':109.4, 'Tempo':65.9,  'Luck':0.010},
    'Siena':        {'Pom_Rank':192,'Reg_Wins':23, 'Reg_Losses':11, 'NetRtg':-2.10, 'ORtg':107.1, 'DRtg':109.2, 'Tempo':64.6,  'Luck':0.005},
    'LIU':          {'Pom_Rank':216,'Reg_Wins':24, 'Reg_Losses':10, 'NetRtg':-3.95, 'ORtg':105.6, 'DRtg':109.6, 'Tempo':67.8,  'Luck':0.104},
    'Kennesaw St.': {'Pom_Rank':163,'Reg_Wins':21, 'Reg_Losses':13, 'NetRtg':0.57,  'ORtg':110.6, 'DRtg':110.1, 'Tempo':71.2,  'Luck':0.009},
    'Queens':       {'Pom_Rank':181,'Reg_Wins':21, 'Reg_Losses':13, 'NetRtg':-1.44, 'ORtg':115.8, 'DRtg':117.2, 'Tempo':69.6,  'Luck':0.067},
    'Troy':         {'Pom_Rank':143,'Reg_Wins':22, 'Reg_Losses':11, 'NetRtg':1.72,  'ORtg':110.7, 'DRtg':109.0, 'Tempo':64.9,  'Luck':0.024},
    'Penn':         {'Pom_Rank':150,'Reg_Wins':18, 'Reg_Losses':11, 'NetRtg':1.47,  'ORtg':107.4, 'DRtg':105.9, 'Tempo':69.0,  'Luck':0.068},
    'Idaho':        {'Pom_Rank':145,'Reg_Wins':21, 'Reg_Losses':14, 'NetRtg':1.53,  'ORtg':108.8, 'DRtg':107.3, 'Tempo':67.7,  'Luck':-0.012},
    'Wright St.':   {'Pom_Rank':140,'Reg_Wins':23, 'Reg_Losses':11, 'NetRtg':2.04,  'ORtg':112.1, 'DRtg':110.0, 'Tempo':67.2,  'Luck':0.009},
    'Tennessee St.':{'Pom_Rank':187,'Reg_Wins':23, 'Reg_Losses':9,  'NetRtg':-1.83, 'ORtg':109.1, 'DRtg':110.9, 'Tempo':70.2,  'Luck':0.070},
    'UMBC':         {'Pom_Rank':185,'Reg_Wins':24, 'Reg_Losses':8,  'NetRtg':-1.67, 'ORtg':108.2, 'DRtg':109.9, 'Tempo':66.2,  'Luck':0.046},
    'Howard':       {'Pom_Rank':207,'Reg_Wins':23, 'Reg_Losses':10, 'NetRtg':-3.19, 'ORtg':103.1, 'DRtg':106.3, 'Tempo':69.0,  'Luck':-0.010},
    'Prairie View A&M':{'Pom_Rank':288,'Reg_Wins':18,'Reg_Losses':17,'NetRtg':-10.69,'ORtg':101.2,'DRtg':111.9, 'Tempo':70.9,  'Luck':0.013},
    'Lehigh':       {'Pom_Rank':284,'Reg_Wins':18, 'Reg_Losses':16, 'NetRtg':-10.37,'ORtg':102.7, 'DRtg':113.1, 'Tempo':66.9,  'Luck':0.081},
    # First Four play-in teams
    'Texas':        {'Pom_Rank':37, 'Reg_Wins':18, 'Reg_Losses':14, 'NetRtg':19.03, 'ORtg':125.0, 'DRtg':105.9, 'Tempo':66.9,  'Luck':-0.083},
    'N.C. State':   {'Pom_Rank':34, 'Reg_Wins':20, 'Reg_Losses':13, 'NetRtg':19.60, 'ORtg':124.1, 'DRtg':104.5, 'Tempo':69.1,  'Luck':-0.029},
    'Miami OH':     {'Pom_Rank':93, 'Reg_Wins':31, 'Reg_Losses':1,  'NetRtg':8.26,  'ORtg':116.8, 'DRtg':108.5, 'Tempo':69.9,  'Luck':0.099},
    'SMU':          {'Pom_Rank':42, 'Reg_Wins':20, 'Reg_Losses':13, 'NetRtg':18.09, 'ORtg':122.9, 'DRtg':104.8, 'Tempo':68.5,  'Luck':-0.043},
}

# ─────────────────────────────────────────────────────────────────────
# 3. DEFINE 2026 BRACKET
# ─────────────────────────────────────────────────────────────────────
# Format: (team_name, seed, region)
# First Four results (predicted before main bracket)
# West: (11) Texas vs N.C. State  ->  winner plays in 11 slot
# Midwest: (11) Miami OH vs SMU   ->  winner plays in 11 slot
# Midwest: (16) UMBC vs Howard    ->  winner plays in 16 slot
# South: (16) Prairie View A&M vs Lehigh -> winner plays in 16 slot

# We'll predict First Four first, then plug winners into bracket
def get_stats(name, seed):
    s = kenpom_2026.get(name, {})
    return {
        'Net Rtg (AdjEM)': s.get('NetRtg', 0),
        'Adj ORtg':        s.get('ORtg', 105),
        'Adj DRtg':        s.get('DRtg', 105),
        'Adj Tempo':       s.get('Tempo', 68),
        'Luck':            s.get('Luck', 0),
        'Pomeroy Rank':    s.get('Pom_Rank', 150),
        'Reg Wins':        s.get('Reg_Wins', 20),
        'Reg Losses':      s.get('Reg_Losses', 12),
        'Seed':            seed,
    }

def predict_matchup(t1_name, t1_seed, t2_name, t2_seed, rnd):
    f = make_matchup_features(get_stats(t1_name, t1_seed), get_stats(t2_name, t2_seed))
    f['round'] = rnd
    fv = pd.DataFrame([f])[feature_cols].fillna(0)
    prob = ensemble.predict_proba(fv)[0][1]
    winner = t1_name if prob >= 0.5 else t2_name
    winner_seed = t1_seed if prob >= 0.5 else t2_seed
    return winner, winner_seed, prob

results = []
def log(rnd_name, t1, s1, t2, s2, rnd_num, region=''):
    w, ws, prob = predict_matchup(t1, s1, t2, s2, rnd_num)
    loser = t2 if w == t1 else t1
    loser_seed = s2 if w == t1 else s1
    win_prob = prob if w == t1 else 1 - prob
    results.append({'Round': rnd_name, 'Region': region,
                    'Team1': t1, 'Seed1': s1, 'Team2': t2, 'Seed2': s2,
                    'Winner': w, 'Winner_Seed': ws,
                    'Win_Prob': round(win_prob, 3)})
    return w, ws

print("\n" + "="*70)
print("  2026 NCAA TOURNAMENT PREDICTIONS")
print("="*70)

# ─── FIRST FOUR ───────────────────────────────────────────────────────
print("\n--- FIRST FOUR ---")
w11w, _ = log('First Four', 'Texas', 11, 'N.C. State', 11, 1, 'West')
w11m, _ = log('First Four', 'Miami OH', 11, 'SMU', 11, 1, 'Midwest')
w16m, _ = log('First Four', 'UMBC', 16, 'Howard', 16, 1, 'Midwest')
w16s, _ = log('First Four', 'Prairie View A&M', 16, 'Lehigh', 16, 1, 'South')

for r in results[-4:]:
    print(f"  ({r['Seed1']}) {r['Team1']} vs ({r['Seed2']}) {r['Team2']}  -->  {r['Winner']} ({r['Win_Prob']:.0%})")

# ─── BUILD BRACKET ────────────────────────────────────────────────────
# Each region: list of (team, seed) in bracket order
# matchup pairs in R64: (1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15)

east = [
    ('Duke', 1), ('Siena', 16),
    ('Ohio St.', 8), ('TCU', 9),
    ('St. John\'s', 5), ('Northern Iowa', 12),
    ('Kansas', 4), ('Cal Baptist', 13),
    ('Louisville', 6), ('South Florida', 11),
    ('Michigan St.', 3), ('Furman', 15),
    ('UCLA', 7), ('North Dakota St.', 14),
    ('Connecticut', 2), ('Wright St.', 14),  # Wright St listed as 14 but should be... let me use as given
]
# Correcting: Wright St is 14 in Midwest not East. East 15 = Furman. Let me rebuild properly.

east = [
    ('Duke', 1),        ('Siena', 16),
    ('Ohio St.', 8),    ('TCU', 9),
    ('St. John\'s', 5), ('Northern Iowa', 12),
    ('Kansas', 4),      ('Cal Baptist', 13),
    ('Louisville', 6),  ('South Florida', 11),
    ('Michigan St.', 3),('Furman', 15),
    ('UCLA', 7),        ('North Dakota St.', 14),
    ('Connecticut', 2), ('Queens', 15),
]
# Note: Queens is 15 in West, not East. Let me use the actual bracket provided:
# East: 1 Duke, 2 UConn, 3 Michigan St, 4 Kansas, 5 St Johns, 6 Louisville,
#       7 UCLA, 8 Ohio St, 9 TCU, 10 UCF, 11 South Florida, 12 Northern Iowa,
#       13 Cal Baptist, 14 NDSU, 15 Furman, 16 Siena

east = [
    ('Duke', 1),         ('Siena', 16),
    ('Ohio St.', 8),     ('TCU', 9),
    ('St. John\'s', 5),  ('Northern Iowa', 12),
    ('Kansas', 4),       ('Cal Baptist', 13),
    ('Louisville', 6),   ('South Florida', 11),
    ('Michigan St.', 3), ('Furman', 15),
    ('UCLA', 7),         ('North Dakota St.', 14),
    ('Connecticut', 2),  ('UCF', 10),  # wait, UCF is 10 in East
]
# Rebuilding cleanly from the bracket provided:
# East: 1 Duke, 16 Siena | 8 Ohio St, 9 TCU | 5 St Johns, 12 NI | 4 Kansas, 13 Cal Baptist
#       6 Louisville, 11 South Florida | 3 Mich St, 15 Furman | 7 UCLA, 10 UCF | 2 UConn, 16 (already Siena used 16)
# UCF is seed 10 in East per listing. Siena is 16.

east = [
    ('Duke', 1),         ('Siena', 16),
    ('Ohio St.', 8),     ('TCU', 9),
    ('St. John\'s', 5),  ('Northern Iowa', 12),
    ('Kansas', 4),       ('Cal Baptist', 13),
    ('Louisville', 6),   ('South Florida', 11),
    ('Michigan St.', 3), ('Furman', 15),
    ('UCLA', 7),         ('UCF', 10),
    ('Connecticut', 2),  ('Wright St.', 14),  # Wright St listed in Midwest... using NDSU 14 for East
]

# Per the bracket as given (I'll be exact):
# East: 1Duke 2UConn 3MichiganSt 4Kansas 5StJohns 6Louisville 7UCLA 8OhioSt 9TCU 10UCF 11SouthFlorida 12NorthernIowa 13CalBaptist 14NDSU 15Furman 16Siena
east = [
    ('Duke', 1),         ('Siena', 16),
    ('Ohio St.', 8),     ('TCU', 9),
    ('St. John\'s', 5),  ('Northern Iowa', 12),
    ('Kansas', 4),       ('Cal Baptist', 13),
    ('Louisville', 6),   ('South Florida', 11),
    ('Michigan St.', 3), ('Furman', 15),
    ('UCLA', 7),         ('UCF', 10),
    ('Connecticut', 2),  ('North Dakota St.', 14),
]

# West: 1Arizona 2Purdue 3Gonzaga 4Arkansas 5Wisconsin 6BYU 7Miami(FL) 8Villanova 9UtahSt 10Missouri 11Texas/NCState(winner) 12HighPoint 13Hawaii 14KennesawSt 15Queens 16LIU
west = [
    ('Arizona', 1),     ('LIU', 16),
    ('Villanova', 8),   ('Utah St.', 9),
    ('Wisconsin', 5),   ('High Point', 12),
    ('Arkansas', 4),    ('Hawaii', 13),
    ('BYU', 6),         (w11w, 11),
    ('Gonzaga', 3),     ('Kennesaw St.', 14),
    ('Miami FL', 7),    ('Missouri', 10),
    ('Purdue', 2),      ('Queens', 15),
]

# Midwest: 1Michigan 2IowaState 3Virginia 4Alabama 5TexasTech 6Tennessee 7Kentucky 8Georgia 9SaintLouis 10SantaClara 11MiamiOH/SMU(winner) 12Akron 13Hofstra 14WrightSt 15TennesseeSt 16UMBC/Howard(winner)
midwest = [
    ('Michigan', 1),    (w16m, 16),
    ('Georgia', 8),     ('Saint Louis', 9),
    ('Texas Tech', 5),  ('Akron', 12),
    ('Alabama', 4),     ('Hofstra', 13),
    ('Tennessee', 6),   (w11m, 11),
    ('Virginia', 3),    ('Wright St.', 14),
    ('Kentucky', 7),    ('Santa Clara', 10),
    ('Iowa St.', 2),    ('Tennessee St.', 15),
]

# South: 1Florida 2Houston 3Illinois 4Nebraska 5Vanderbilt 6NorthCarolina 7SaintMarys 8Clemson 9Iowa 10TexasA&M 11VCU 12McNeese 13Troy 14Penn 15Idaho 16PVA&M/Lehigh(winner)
south = [
    ('Florida', 1),     (w16s, 16),
    ('Clemson', 8),     ('Iowa', 9),
    ('Vanderbilt', 5),  ('McNeese', 12),
    ('Nebraska', 4),    ('Troy', 13),
    ('North Carolina', 6), ('VCU', 11),
    ('Illinois', 3),    ('Penn', 14),
    ('Saint Mary\'s', 7), ('Texas A&M', 10),
    ('Houston', 2),     ('Idaho', 15),
]

def sim_region(region_name, bracket):
    """Simulate all rounds within a region, return Final Four team."""
    print(f"\n{'='*60}")
    print(f"  {region_name.upper()} REGION")
    print(f"{'='*60}")
    
    round_names = {1: 'First Round (R64)', 2: 'Round of 32', 3: 'Sweet 16', 4: 'Elite Eight'}
    current = bracket[:]  # list of (team, seed) pairs
    rnd = 1

    while len(current) > 1:
        print(f"\n  --- {round_names[rnd]} ---")
        next_round = []
        for i in range(0, len(current), 2):
            t1, s1 = current[i]
            t2, s2 = current[i+1]
            w, ws = log(round_names[rnd], t1, s1, t2, s2, rnd, region_name)
            print(f"    ({s1}) {t1:<22} vs ({s2}) {t2:<22}  -->  {w}")
            next_round.append((w, ws))
        current = next_round
        rnd += 1
    
    champion, champ_seed = current[0]
    print(f"\n  >> {region_name} Champion: ({champ_seed}) {champion}")
    return champion, champ_seed

# ─── SIMULATE ALL 4 REGIONS ──────────────────────────────────────────
e_winner, e_seed = sim_region('East', east)
w_winner, w_seed = sim_region('West', west)
m_winner, m_seed = sim_region('Midwest', midwest)
s_winner, s_seed = sim_region('South', south)

# ─── FINAL FOUR ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FINAL FOUR")
print(f"{'='*60}")

# Traditional bracket: East vs West, Midwest vs South
ff1_w, ff1_s = log('Final Four', e_winner, e_seed, w_winner, w_seed, 5)
ff2_w, ff2_s = log('Final Four', m_winner, m_seed, s_winner, s_seed, 5)
print(f"\n  ({e_seed}) {e_winner} vs ({w_seed}) {w_winner}  -->  {ff1_w}")
print(f"  ({m_seed}) {m_winner} vs ({s_seed}) {s_winner}  -->  {ff2_w}")

# ─── CHAMPIONSHIP ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  NATIONAL CHAMPIONSHIP")
print(f"{'='*60}")
champ, champ_seed = log('Championship', ff1_w, ff1_s, ff2_w, ff2_s, 6)
_, _, champ_prob = predict_matchup(ff1_w, ff1_s, ff2_w, ff2_s, 6)
champ_win_prob = champ_prob if champ == ff1_w else 1 - champ_prob
print(f"\n  ({ff1_s}) {ff1_w} vs ({ff2_s}) {ff2_w}  -->  {champ} ({champ_win_prob:.0%})")
print(f"\n  *** 2026 PREDICTED CHAMPION: {champ} ***")

# ─── SAVE RESULTS ────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv('/home/claude/predictions_2026.csv', index=False)
print(f"\nSaved {len(results_df)} game predictions.")
print(results_df[['Round','Region','Winner','Winner_Seed','Win_Prob']].to_string(index=False))
