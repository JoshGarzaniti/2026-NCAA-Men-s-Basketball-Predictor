import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

##Loading Train Data
df = pd.read_excel('/mnt/user-data/uploads/ncaa_tournament_dataset.xlsx', sheet_name='Tournament Dataset')
print(f"Loaded {df.shape[0]} rows, years: {sorted(df['Year'].unique())}")

ROUND_ORDER = {
    'First Round': 1,
    'Round of 32': 2,
    'Sweet 16': 3,
    'Elite Eight': 4,
    'Final Four': 5,
    'Champion': 6,
    '—': 0   # first-four losers / did not advance
}
df['Round_Num'] = df['Round Reached'].map(ROUND_ORDER)
df['Tourn_Wins'] = pd.to_numeric(df['Tourn Wins'], errors='coerce').fillna(0).astype(int)

#Build in 2025 data (both the team stats and the tournament results)
kenpom_2025_raw = """
1,Duke,ACC,35,4,1,39.29,130.1,90.85,66.02
2,Houston,B12,35,5,2,36.59,123.41,86.81,61.93
3,Florida,SEC,36,4,3,36.46,128.22,91.86,70.15
4,Auburn,SEC,32,6,4,35.05,127.23,92.19,68.21
5,Tennessee,SEC,30,8,5,30.93,120.71,89.73,63.73
6,Alabama,SEC,28,9,6,30.34,126.84,96.52,75.31
7,Michigan St.,B10,30,7,7,28.48,118.82,90.44,67.51
8,Gonzaga,WCC,26,9,8,28.01,124.76,96.72,70.54
9,Texas Tech,B12,28,9,9,27.92,125.45,97.53,66.12
10,Maryland,B10,27,9,10,27.07,118.92,91.97,69.66
11,Iowa St.,B12,25,10,11,26.65,119.81,93.21,68.81
12,Wisconsin,B10,27,10,12,26.57,122.91,96.32,68.01
13,Arizona,B12,24,13,13,26.35,123.71,97.33,70.34
14,St. John's,BE,31,5,14,26.06,114.06,88.02,70.05
15,Purdue,B10,24,12,15,25.36,124.67,99.25,65.13
16,Kentucky,SEC,24,12,16,24.54,123.71,99.25,70.93
17,Illinois,B10,22,13,17,24.32,121.91,97.64,71.51
18,Texas A&M,SEC,23,11,18,23.66,116.54,92.91,66.82
19,Missouri,SEC,22,12,19,23.57,124.48,100.86,68.61
20,UCLA,B10,23,11,20,23.46,117.33,93.81,65.03
21,Michigan,B10,27,10,21,23.45,116.54,93.11,70.15
22,Ole Miss,SEC,24,12,22,23.34,119.12,95.82,68.11
23,Clemson,ACC,27,7,23,23.22,118.22,95.01,64.63
24,Kansas,B12,21,13,24,22.94,115.95,93.01,68.61
25,Saint Mary's,WCC,29,6,25,22.73,114.76,92.08,61.73
26,BYU,B12,26,10,26,22.31,124.19,101.88,67.61
27,Oregon,B10,25,10,27,21.85,117.63,95.82,68.11
28,Louisville,ACC,27,8,28,21.73,118.13,96.32,69.19
29,Marquette,BE,23,11,29,21.01,117.43,96.42,67.71
30,Baylor,B12,20,15,30,20.83,121.31,100.46,64.63
31,North Carolina,ACC,23,14,31,20.33,119.22,98.94,70.73
32,Connecticut,BE,24,11,32,20.18,121.61,101.47,64.03
33,Mississippi St.,SEC,21,13,33,20.02,118.92,98.84,68.31
34,VCU,A10,28,7,34,19.53,116.05,96.52,66.32
35,Creighton,BE,25,11,35,19.44,117.83,98.34,67.31
36,Arkansas,SEC,22,14,36,19.33,114.46,95.11,69.46
37,Ohio St.,B10,17,15,37,19.08,118.23,99.15,67.31
38,Georgia,SEC,20,13,38,18.57,115.35,96.83,66.72
39,UC San Diego,BW,30,5,39,18.25,115.06,96.73,65.82
40,Oklahoma,SEC,20,14,40,17.95,118.22,100.36,68.81
41,New Mexico,MWC,27,8,41,17.78,112.97,95.11,72.57
42,Colorado St.,MWC,26,10,42,17.61,116.84,99.25,66.02
43,Xavier,BE,22,12,43,17.19,116.24,99.04,69.28
44,Northwestern,B10,17,16,44,17.02,114.36,97.33,65.22
45,Indiana,B10,19,13,45,16.98,114.26,97.23,68.21
46,Nebraska,B10,21,14,46,16.98,114.76,97.74,67.81
47,Texas,SEC,19,16,47,16.90,117.33,100.46,67.61
48,Vanderbilt,SEC,20,13,48,16.23,117.83,101.67,69.56
49,Boise St.,MWC,26,11,49,16.04,116.93,100.97,66.02
50,SMU,ACC,24,11,50,15.42,116.24,100.86,68.99
51,San Diego St.,MWC,21,10,51,15.13,109.81,94.61,66.32
52,Drake,MVC,31,4,52,15.12,112.97,97.84,59.43
53,West Virginia,B12,19,13,53,15.11,109.01,93.91,64.33
54,Memphis,Amer,29,6,54,15.10,113.47,98.34,71.02
55,Cincinnati,B12,19,16,55,14.86,110.61,95.82,66.12
56,Villanova,BE,21,15,56,14.74,118.62,103.81,63.43
57,USC,B10,17,18,57,14.39,116.64,102.28,68.61
58,Penn St.,B10,16,15,58,14.36,115.25,100.86,70.15
59,Santa Clara,WCC,21,13,59,14.34,116.34,102.08,69.28
60,McNeese,Slnd,28,7,60,14.24,114.16,99.85,65.92
61,Utah St.,MWC,26,8,61,14.00,119.82,105.81,67.61
62,Iowa,B10,17,16,62,13.56,120.18,106.51,69.86
63,Pittsburgh,ACC,17,15,63,13.50,115.75,102.29,66.52
64,Liberty,CUSA,28,7,64,12.46,111.89,99.45,65.62
65,Kansas St.,B12,16,17,65,12.17,109.41,97.23,67.12
66,San Francisco,WCC,25,10,66,12.15,111.89,99.75,68.11
67,UCF,B12,20,17,67,11.95,116.05,104.11,72.21
68,UC Irvine,BW,32,7,68,11.75,106.91,95.12,69.37
69,South Carolina,SEC,12,20,69,11.51,110.61,99.14,65.52
70,Rutgers,B10,15,17,70,11.50,115.75,104.21,68.91
71,North Texas,Amer,27,9,71,11.30,109.71,98.44,61.03
72,Wake Forest,ACC,21,11,72,11.18,108.31,97.13,67.31
73,Arizona St.,B12,13,20,73,11.03,112.48,101.47,68.99
74,Yale,Ivy,22,8,74,11.02,115.25,104.21,67.71
75,Butler,BE,15,20,75,10.54,117.23,106.71,68.01
76,Utah,B12,16,17,76,10.50,112.58,102.08,69.18
77,Dayton,A10,23,11,77,9.98,115.75,105.71,66.22
78,Nevada,MWC,17,16,78,9.76,112.87,103.09,64.03
79,George Mason,A10,27,9,79,9.46,106.21,96.73,64.93
80,Oregon St.,WCC,20,13,80,9.37,116.24,106.81,63.94
81,Stanford,ACC,21,14,81,9.26,112.58,103.29,66.32
82,High Point,BSth,29,6,82,9.15,118.42,109.22,66.52
83,Minnesota,B10,15,17,83,9.15,110.61,101.57,62.93
84,Arkansas St.,SB,25,11,84,9.13,111.49,102.39,70.25
85,Saint Joseph's,A10,22,13,85,9.07,109.71,100.66,68.21
86,TCU,B12,16,16,86,8.82,105.82,97.03,66.72
87,Colorado,B12,14,21,87,8.73,108.61,99.85,68.01
88,LSU,SEC,14,18,88,8.72,109.21,100.56,68.21
89,Lipscomb,ASun,25,10,89,8.55,112.08,103.51,66.22
90,Georgetown,BE,18,16,90,8.49,108.71,100.25,68.51
91,Florida St.,ACC,17,15,91,8.20,108.81,100.66,69.76
92,Bradley,MVC,28,9,92,7.89,111.59,103.61,65.92
93,UNLV,MWC,18,15,93,7.21,107.51,100.35,66.02
94,Chattanooga,SC,29,9,94,7.14,116.44,109.32,66.82
95,Troy,SB,23,11,95,6.95,107.41,100.46,66.52
96,Providence,BE,12,20,96,6.93,112.18,105.21,66.52
97,Oklahoma St.,B12,17,18,97,6.91,108.01,101.17,72.01
98,Notre Dame,ACC,15,18,98,6.90,111.39,104.41,64.93
99,UAB,Amer,24,13,99,6.70,116.34,109.62,69.56
100,Akron,MAC,28,7,100,6.47,113.67,107.11,71.71
101,Grand Canyon,WAC,26,8,101,6.29,107.41,101.17,71.41
102,Northern Iowa,MVC,20,13,102,6.09,109.61,103.51,65.42
103,Georgia Tech,ACC,17,17,103,5.87,107.21,101.37,68.91
104,Washington,B10,13,18,104,5.79,109.61,103.81,67.91
105,UNC Wilmington,CAA,27,8,105,5.28,112.28,106.91,65.92
106,Virginia,ACC,15,17,106,5.02,110.61,105.51,61.33
107,Loyola Chicago,A10,25,12,107,4.94,108.11,103.19,67.12
108,Jacksonville St.,CUSA,23,13,108,4.92,109.81,104.91,65.13
109,South Alabama,SB,21,11,109,4.49,107.71,103.29,63.33
110,California,ACC,14,19,110,4.43,111.39,106.91,67.71
111,Kent St.,MAC,24,12,111,4.30,106.21,101.98,66.72
112,Utah Valley,WAC,25,9,112,4.25,106.71,102.49,68.41
113,North Alabama,ASun,24,11,113,4.24,111.19,106.91,66.92
114,Wofford,SC,19,16,114,4.24,113.96,109.72,64.33
115,Syracuse,ACC,14,19,115,4.22,110.31,106.11,68.41
116,Florida Atlantic,Amer,18,16,116,4.13,112.68,108.52,69.09
117,CSUN,BW,22,11,117,4.09,109.01,104.91,72.58
118,Samford,SC,22,11,118,4.03,112.87,108.82,71.12
119,Middle Tennessee,CUSA,22,12,119,3.98,109.21,105.21,68.51
120,Washington St.,WCC,19,15,120,3.86,111.98,108.01,70.83
121,Saint Louis,A10,19,15,121,3.85,109.11,105.31,67.81
122,DePaul,BE,14,20,122,3.83,109.21,105.41,67.41
123,St. Bonaventure,A10,22,12,123,3.82,105.32,101.47,64.83
124,Belmont,MVC,22,11,124,3.64,112.67,109.02,70.73
125,N.C. State,ACC,12,19,125,3.63,107.61,103.91,65.52
126,Northern Colorado,BSky,25,10,126,3.63,112.28,108.62,68.99
127,Louisiana Tech,CUSA,20,12,127,3.53,108.91,105.41,64.73
128,George Washington,A10,21,13,128,3.39,107.11,103.71,67.41
129,Cornell,Ivy,18,11,129,3.37,115.06,111.72,71.12
130,St. Thomas,Sum,24,10,130,3.28,113.77,110.42,68.11
131,Kennesaw St.,CUSA,19,14,131,3.02,107.11,104.11,70.54
132,Illinois St.,MVC,22,14,132,2.92,112.77,109.82,66.22
133,Robert Morris,Horz,26,9,133,2.90,107.51,104.61,68.11
134,New Mexico St.,CUSA,17,15,134,2.82,104.32,101.57,65.13
135,North Dakota St.,Sum,21,11,135,2.62,115.45,112.72,65.13
136,Wichita St.,Amer,19,15,136,2.51,107.21,104.71,68.91
137,South Dakota St.,Sum,20,12,137,2.50,109.71,107.21,69.28
138,Murray St.,MVC,16,17,138,2.27,106.81,104.51,65.03
139,Furman,SC,25,10,139,2.26,110.31,108.11,66.12
140,Duquesne,A10,13,19,140,2.16,105.92,103.81,65.42
141,Milwaukee,Horz,21,11,141,2.11,107.51,105.41,68.99
142,East Tennessee St.,SC,19,13,142,1.89,106.81,104.91,65.32
143,UNC Greensboro,SC,20,12,143,1.84,107.31,105.51,64.03
144,Tulane,Amer,19,15,144,1.42,108.31,106.81,67.12
145,Seattle,WAC,14,18,145,1.37,104.02,102.69,66.22
146,Miami OH,MAC,25,9,146,1.09,108.11,107.01,69.28
147,Davidson,A10,17,16,147,1.08,109.51,108.52,66.12
148,UC Santa Barbara,BW,21,13,148,1.07,108.71,107.61,65.72
149,Bryant,AE,23,12,149,0.92,106.81,105.81,72.76
150,Texas A&M CC,Slnd,20,14,150,0.81,106.41,105.61,67.61
151,Radford,BSth,20,13,151,0.62,110.51,109.92,63.03
152,Towson,CAA,22,11,152,0.58,107.01,106.41,63.83
153,Charleston,CAA,24,9,153,0.48,106.11,105.61,70.25
154,Cleveland St.,Horz,23,13,154,0.45,105.02,104.51,65.62
155,Loyola Marymount,WCC,17,15,155,0.39,104.22,103.81,67.41
156,Purdue Fort Wayne,Horz,19,13,156,0.37,109.31,109.02,70.05
157,Rhode Island,A10,18,13,157,0.35,108.01,107.61,70.63
158,Lamar,Slnd,20,13,158,0.31,102.42,102.18,65.62
159,UC Riverside,BW,21,13,159,0.28,109.81,109.52,67.31
160,James Madison,SB,20,12,160,0.24,110.69,110.42,65.13
161,Western Kentucky,CUSA,17,15,161,0.21,103.52,103.31,71.91
162,UTEP,CUSA,18,15,162,0.18,105.32,105.11,69.09
163,Montana,BSky,25,10,163,0.07,110.69,110.62,67.31
164,Virginia Tech,ACC,13,19,164,-0.02,106.51,106.51,65.32
165,Nebraska Omaha,Sum,22,13,165,-0.06,111.09,111.12,67.71
166,Winthrop,BSth,23,11,166,-0.13,108.41,108.52,72.94
167,Nicholls,Slnd,20,13,167,-0.24,106.01,106.31,68.31
168,Marshall,SB,20,13,168,-0.30,106.51,106.81,68.91
169,San Jose St.,MWC,15,20,169,-0.45,107.21,107.71,66.72
170,Appalachian St.,SB,17,14,170,-0.45,101.22,101.78,63.43
171,Illinois Chicago,MVC,17,14,171,-0.52,107.31,107.81,69.37
172,Cal Baptist,WAC,17,15,172,-0.63,105.72,106.41,65.92
173,Temple,Amer,17,15,173,-0.68,110.11,110.82,68.91
174,Oakland,Horz,16,18,174,-0.91,107.41,108.32,64.13
175,Sam Houston St.,CUSA,13,19,175,-1.05,110.41,111.52,67.81
176,East Carolina,Amer,19,14,176,-1.11,107.11,108.22,66.82
177,Princeton,Ivy,19,11,177,-1.45,105.42,106.91,66.72
178,Norfolk St.,MEAC,24,11,178,-1.63,107.51,109.12,66.52
179,Florida Gulf Coast,ASun,19,15,179,-1.81,107.21,109.02,65.52
180,Central Connecticut,NEC,25,7,180,-1.87,99.92,101.78,66.02
181,Boston College,ACC,12,19,181,-2.00,105.42,107.41,66.92
182,Cal Poly,BW,16,19,182,-2.13,109.41,111.52,74.73
183,Drexel,CAA,18,15,183,-2.19,104.02,106.21,63.23
184,Merrimack,MAAC,18,15,184,-2.21,99.92,102.18,64.73
185,Wyoming,MWC,12,20,185,-2.23,103.52,105.81,64.83
186,Southeastern Louisiana,Slnd,18,14,186,-2.23,105.02,107.21,67.12
187,Eastern Kentucky,ASun,18,14,187,-2.24,109.51,111.82,67.81
188,Rice,Amer,13,19,188,-2.26,106.51,108.71,65.52
189,Montana St.,BSky,15,18,189,-2.28,105.22,107.51,66.12
190,Southern Illinois,MVC,14,19,190,-2.39,106.01,108.42,68.11
191,UTSA,Amer,12,19,191,-2.39,107.41,109.82,70.15
192,South Carolina St.,MEAC,20,13,192,-2.49,102.62,105.11,69.47
193,Miami FL,ACC,7,24,193,-2.61,113.57,116.13,67.71
194,Texas St.,SB,16,16,194,-2.77,109.71,112.42,66.22
195,Ohio,MAC,16,16,195,-2.81,106.32,109.12,70.54
196,South Florida,Amer,13,19,196,-2.84,103.62,106.51,68.51
197,Indiana St.,MVC,14,18,197,-2.85,109.41,112.22,72.49
198,Central Michigan,MAC,14,17,198,-2.85,105.52,108.42,67.41
199,Youngstown St.,Horz,21,13,199,-2.87,102.42,105.31,69.47
200,Jacksonville,ASun,19,14,200,-2.89,101.52,104.41,67.81
"""

lines = [l.strip() for l in kenpom_2025_raw.strip().split('\n') if l.strip()]
kenpom_2025 = []
for line in lines:
    parts = line.split(',')
    if len(parts) >= 10:
        try:
            kenpom_2025.append({
                'Pom_Rank': int(parts[0]),
                'Team': parts[1].strip(),
                'Conference': parts[2].strip(),
                'Reg_Wins': int(parts[3]),
                'Reg_Losses': int(parts[4]),
                'NetRtg': float(parts[6]),
                'ORtg': float(parts[7]),
                'DRtg': float(parts[8]),
                'Tempo': float(parts[9]),
            })
        except:
            pass

kp2025 = pd.DataFrame(kenpom_2025)
print(f"KenPom 2025: {len(kp2025)} teams loaded")

#Put together a bracket with context of the 2025 tournament
seeds_2025 = {
    # Region 1 (East?) - Duke's region
    'Duke': 1, 'Mount St. Mary\'s': 16, 'Mississippi St.': 8, 'Baylor': 9,
    'Iowa St.': 5, 'Lipscomb': 12, 'BYU': 6, 'VCU': 11,
    'Arizona': 4, 'Akron': 13, 'Illinois': 3, 'Texas': 11,
    'Alabama': 2, 'Robert Morris': 15,
    # Region 2 - Auburn
    'Auburn': 1, 'Alabama State': 16, 'Creighton': 9, 'Louisville': 8,
    'Michigan': 5, 'UC San Diego': 12, 'Missouri': 6, 'Drake': 11,
    'Texas A&M': 4, 'Yale': 13, 'Wisconsin': 3, 'Montana': 14,
    'Tennessee': 2, 'Wofford': 15,
    # Region 3 - Houston
    'Houston': 1, 'SIU Edwardsville': 16, 'Georgia': 9, 'Gonzaga': 8,
    'Clemson': 5, 'McNeese': 12, 'Ole Miss': 6, 'North Carolina': 11,
    'Maryland': 4, 'Grand Canyon': 13, 'Kentucky': 3, 'Troy': 14,
    'Michigan St.': 2, 'Bryant': 15,
    # Region 4 - Florida
    'Florida': 1, 'Norfolk St.': 16, 'UConn': 8, 'Oklahoma': 9,
    'Oregon': 5, 'Liberty': 12, 'Saint Mary\'s': 7, 'Vanderbilt': 10,
    'Purdue': 4, 'High Point': 13, 'Texas Tech': 3, 'UNC Wilmington': 14,
    'St. John\'s': 2, 'Nebraska Omaha': 15,
    # Extra (first four teams that appear in results)
    'San Diego St.': 11, 'Xavier': 11, 'UCLA': 7, 'Utah St.': 10,
    'Arkansas': 10, 'Kansas': 7, 'New Mexico': 10, 'Marquette': 7,
    'Colorado St.': 12, 'Memphis': 5,
}

# Actual 2025 results from context doc
results_2025_raw = [
    # First Round R64
    ('Creighton',9,'Louisville',8,1,'R64'),
    ('Purdue',4,'High Point',13,1,'R64'),
    ('Wisconsin',3,'Montana',14,1,'R64'),
    ('Houston',1,'SIU Edwardsville',16,1,'R64'),
    ('Auburn',1,'Alabama State',16,1,'R64'),
    ('McNeese',12,'Clemson',5,0,'R64'),
    ('BYU',6,'VCU',11,1,'R64'),
    ('Gonzaga',8,'Georgia',9,1,'R64'),
    ('Tennessee',2,'Wofford',15,1,'R64'),
    ('Arkansas',10,'Kansas',7,1,'R64'),
    ('Texas A&M',4,'Yale',13,1,'R64'),
    ('Drake',11,'Missouri',6,0,'R64'),
    ('UCLA',7,'Utah St.',10,1,'R64'),
    ('St. John\'s',2,'Nebraska Omaha',15,1,'R64'),
    ('Michigan',5,'UC San Diego',12,1,'R64'),
    ('Texas Tech',3,'UNC Wilmington',14,1,'R64'),
    ('Baylor',9,'Mississippi St.',8,1,'R64'),
    ('Alabama',2,'Robert Morris',15,1,'R64'),
    ('Iowa St.',3,'Lipscomb',14,1,'R64'),
    ('Colorado St.',12,'Memphis',5,0,'R64'),
    ('Duke',1,'Mount St. Mary\'s',16,1,'R64'),
    ('Saint Mary\'s',7,'Vanderbilt',10,1,'R64'),
    ('Ole Miss',6,'North Carolina',11,1,'R64'),
    ('Maryland',4,'Grand Canyon',13,1,'R64'),
    ('Florida',1,'Norfolk St.',16,1,'R64'),
    ('Kentucky',3,'Troy',14,1,'R64'),
    ('New Mexico',10,'Marquette',7,0,'R64'),
    ('Arizona',4,'Akron',13,1,'R64'),
    ('UConn',8,'Oklahoma',9,1,'R64'),
    ('Illinois',3,'Xavier',6,1,'R64'),  # actually seed 11 Xavier
    ('Michigan St.',2,'Bryant',15,1,'R64'),
    ('Oregon',5,'Liberty',12,1,'R64'),
    # R32
    ('Purdue',4,'McNeese',12,1,'R32'),
    ('Arkansas',10,'St. John\'s',2,0,'R32'),
    ('Michigan',5,'Texas A&M',4,1,'R32'),
    ('Texas Tech',3,'Drake',11,1,'R32'),
    ('Auburn',1,'Creighton',9,1,'R32'),
    ('BYU',6,'Wisconsin',3,1,'R32'),
    ('Houston',1,'Gonzaga',8,1,'R32'),
    ('Tennessee',2,'UCLA',7,1,'R32'),
    ('Florida',1,'UConn',8,1,'R32'),
    ('Duke',1,'Baylor',9,1,'R32'),
    ('Kentucky',3,'Illinois',6,1,'R32'),
    ('Alabama',2,'Saint Mary\'s',7,1,'R32'),
    ('Maryland',4,'Colorado St.',12,1,'R32'),
    ('Ole Miss',6,'Iowa St.',3,0,'R32'),
    ('Michigan St.',2,'New Mexico',10,1,'R32'),
    ('Arizona',4,'Oregon',5,1,'R32'),
    # S16
    ('Alabama',2,'BYU',6,1,'S16'),
    ('Florida',1,'Maryland',4,1,'S16'),
    ('Duke',1,'Arizona',4,1,'S16'),
    ('Texas Tech',3,'Arkansas',10,1,'S16'),
    ('Michigan St.',2,'Ole Miss',6,1,'S16'),
    ('Tennessee',2,'Kentucky',3,1,'S16'),
    ('Auburn',1,'Michigan',5,1,'S16'),
    ('Houston',1,'Purdue',4,1,'S16'),
    # E8
    ('Florida',1,'Texas Tech',3,1,'E8'),
    ('Duke',1,'Alabama',2,1,'E8'),
    ('Houston',1,'Tennessee',2,1,'E8'),
    ('Auburn',1,'Michigan St.',2,1,'E8'),
    # F4
    ('Florida',1,'Auburn',1,1,'F4'),
    ('Houston',1,'Duke',1,1,'F4'),
    # NCG
    ('Florida',1,'Houston',1,1,'NCG'),
]

#Load in the matchup and training data from 2002 to 2024

def round_label_to_num(r):
    m = {'First Round':1,'Round of 32':2,'Sweet 16':3,'Elite Eight':4,'Final Four':5,'Champion':6,'—':0}
    return m.get(r,0)

df['Round_Num'] = df['Round Reached'].map(lambda r: round_label_to_num(r))

# For each year, reconstruct matchups: seed 1 vs 16, 8 vs 9, etc in R64
# Then advance winners and track
SEED_MATCHUPS_R64 = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]

def build_matchups_from_year(year_df):
    """Build pairwise matchup records from a tournament year."""
    matchups = []
    teams = year_df.set_index('Team').to_dict('index')
    
    # Map: (team) -> round_num reached
    def rounds_won(team):
        if team in teams:
            return teams[team]['Round_Num']
        return 0
    
    # We'll use the 'Tourn Wins' and 'Round Reached' to infer who beat whom
    # Strategy: for each round, team A beat team B if A advanced further
    # We create matchups by pairing teams that could have met
    # The cleanest approach: use seed bracket position
    
    # Group by seed to get all teams
    seed_teams = {}
    for t, row in teams.items():
        s = int(row.get('Seed', 0)) if pd.notna(row.get('Seed', 0)) else 0
        if s not in seed_teams:
            seed_teams[s] = []
        seed_teams[s].append(t)
    
    return teams, seed_teams

def make_matchup_features(t1_stats, t2_stats):
    """Create feature vector for a matchup. t1 is team 1, t2 is team 2."""
    feats = {}
    cols = ['Net Rtg (AdjEM)', 'Adj ORtg', 'Adj DRtg', 'Adj Tempo', 'Luck',
            'Pomeroy Rank', 'Reg Wins', 'Reg Losses', 'Seed']
    
    for c in cols:
        v1 = t1_stats.get(c, 0) or 0
        v2 = t2_stats.get(c, 0) or 0
        try:
            v1, v2 = float(v1), float(v2)
        except:
            v1, v2 = 0, 0
        safe_c = c.replace(' ', '_').replace('(', '').replace(')', '')
        feats[f't1_{safe_c}'] = v1
        feats[f't2_{safe_c}'] = v2
        feats[f'diff_{safe_c}'] = v1 - v2
    
    # Seed difference
    s1 = float(t1_stats.get('Seed', 8) or 8)
    s2 = float(t2_stats.get('Seed', 8) or 8)
    feats['seed_diff'] = s1 - s2
    feats['higher_seed_is_t1'] = 1 if s1 < s2 else 0  # lower seed # = better
    
    # Win% 
    w1 = float(t1_stats.get('Reg Wins', 0) or 0)
    l1 = float(t1_stats.get('Reg Losses', 1) or 1)
    w2 = float(t2_stats.get('Reg Wins', 0) or 0)
    l2 = float(t2_stats.get('Reg Losses', 1) or 1)
    feats['t1_win_pct'] = w1 / (w1 + l1 + 1e-9)
    feats['t2_win_pct'] = w2 / (w2 + l2 + 1e-9)
    feats['diff_win_pct'] = feats['t1_win_pct'] - feats['t2_win_pct']
    
    return feats

# Build training matchups from historical data
# We reconstruct who played whom each round from seeds and advancement
# Canonical bracket: 4 regions of 16 (post-2011 has first four too)

CANONICAL_BRACKET = {
    'R64': [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)],
    'R32': [(1,8),(5,4),(6,3),(7,2)],  # winners of pairs above
    'S16': [(1,5),(6,2)],  # by region quadrant
    'E8':  [(1,2)],
    'F4':  [(1,2)],  # cross-bracket
    'NCG': [(1,2)]
}

round_name_map = {
    'First Round': 1, 'Round of 32': 2, 'Sweet 16': 3,
    'Elite Eight': 4, 'Final Four': 5, 'Champion': 6
}

all_matchup_records = []

for year in df['Year'].unique():
    ydf = df[df['Year'] == year].copy()
    ydf = ydf.dropna(subset=['Seed'])
    ydf['Seed'] = ydf['Seed'].astype(int)
    
    # Build dict: seed -> list of teams
    by_seed = {}
    for _, row in ydf.iterrows():
        s = int(row['Seed'])
        if s not in by_seed:
            by_seed[s] = []
        by_seed[s].append(row)
    
    teams_dict = {row['Team']: row for _, row in ydf.iterrows()}
    
    # For R64: reconstruct bracket matchups
    # We pair seed 1 vs 16, 8 vs 9, etc. within each "region"
    # Each seed appears 4 times (4 regions), so we need to pair them in groups of 4
    
    # Sort each seed group by Round_Num descending to find which team went furthest
    for seed_pair in [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]:
        s1, s2 = seed_pair
        teams_s1 = sorted(by_seed.get(s1, []), key=lambda r: -r['Round_Num'])
        teams_s2 = sorted(by_seed.get(s2, []), key=lambda r: -r['Round_Num'])
        
        # Pair them up: 4 matchups per year for standard seeds (16 teams per region × 4 regions)
        # Actually each seed has up to 4 teams. Pair them 1:1 by index
        n = min(len(teams_s1), len(teams_s2))
        for i in range(n):
            t1 = teams_s1[i]
            t2 = teams_s2[i]
            
            # winner = team that advanced further
            r1, r2 = t1['Round_Num'], t2['Round_Num']
            if r1 > 0 or r2 > 0:
                winner = 1 if r1 > r2 else 0  # 1 = t1 won
                if r1 == r2:
                    continue  # tie/ambiguous, skip
                
                feats = make_matchup_features(t1.to_dict(), t2.to_dict())
                feats['year'] = year
                feats['round'] = 1  # R64
                feats['winner'] = winner
                all_matchup_records.append(feats)
                
                # Also add flipped version
                feats_flip = make_matchup_features(t2.to_dict(), t1.to_dict())
                feats_flip['year'] = year
                feats_flip['round'] = 1
                feats_flip['winner'] = 1 - winner
                all_matchup_records.append(feats_flip)

# Add higher-round matchups: if team1 reached round R+1 and team2 reached round R
# they likely played in round R
# For rounds 2-6, infer matchups from bracket advancement
# Simpler: for each round level, take all teams that "won" that round vs teams that lost in that round
# We do cross-seed matchups based on advancement

for year in df['Year'].unique():
    ydf = df[df['Year'] == year].copy()
    ydf = ydf.dropna(subset=['Seed'])
    ydf['Seed'] = ydf['Seed'].astype(int)
    teams_dict = {row['Team']: row.to_dict() for _, row in ydf.iterrows()}
    
    for rnd in range(2, 7):  # R32 through NCG
        # Winners at this round: teams that reached at least this round
        winners = [t for t, d in teams_dict.items() if d['Round_Num'] >= rnd]
        losers_at_rnd = [t for t, d in teams_dict.items() if d['Round_Num'] == rnd-1]
        
        # Pair winners with plausible opponents (losers from same round)
        # Match by seed proximity within region
        for w_team in winners:
            w = teams_dict[w_team]
            w_seed = int(w['Seed'])
            
            # Find likely opponent: loser with complementary seed from same quadrant
            # Quadrant groupings by seed: (1,2,3,4,5,6,7,8) per half-bracket
            # Approximate: just find closest seed loser
            best_opp = None
            best_dist = 999
            for l_team in losers_at_rnd:
                l = teams_dict[l_team]
                l_seed = int(l['Seed'])
                dist = abs(w_seed - l_seed)
                if dist < best_dist:
                    best_dist = dist
                    best_opp = l_team
            
            if best_opp and best_dist <= 8:
                l = teams_dict[best_opp]
                feats = make_matchup_features(w, l)
                feats['year'] = year
                feats['round'] = rnd
                feats['winner'] = 1
                all_matchup_records.append(feats)
                
                feats_flip = make_matchup_features(l, w)
                feats_flip['year'] = year
                feats_flip['round'] = rnd
                feats_flip['winner'] = 0
                all_matchup_records.append(feats_flip)

matchup_df = pd.DataFrame(all_matchup_records)
print(f"Training matchups: {len(matchup_df)}")
print(matchup_df.head(2))

#Fit a random forest model around the data
feature_cols = [c for c in matchup_df.columns if c not in ['year','winner']]
X = matchup_df[feature_cols].fillna(0)
y = matchup_df['winner']

# GBM ensemble
gbm = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                             random_state=42)

from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier([('gbm', gbm), ('rf', rf)], voting='soft')
ensemble.fit(X, y)

cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

#Put in the 2025 test data
seeds_2025_v2 = {
    'Duke':1,'Mount St. Mary\'s':16,'Mississippi St.':8,'Baylor':9,
    'Iowa St.':5,'Lipscomb':14,'BYU':6,'VCU':11,
    'Arizona':4,'Akron':13,'Illinois':3,'Texas':11,
    'Alabama':2,'Robert Morris':15,
    'Auburn':1,'Alabama State':16,'Creighton':9,'Louisville':8,
    'Michigan':5,'UC San Diego':12,'Missouri':6,'Drake':11,
    'Texas A&M':4,'Yale':13,'Wisconsin':3,'Montana':14,
    'Tennessee':2,'Wofford':15,
    'Houston':1,'SIU Edwardsville':16,'Georgia':9,'Gonzaga':8,
    'Clemson':5,'McNeese':12,'Ole Miss':6,'North Carolina':11,
    'Maryland':4,'Grand Canyon':13,'Kentucky':3,'Troy':14,
    'Michigan St.':2,'Bryant':15,
    'Florida':1,'Norfolk St.':16,'UConn':8,'Oklahoma':9,
    'Oregon':5,'Liberty':12,'Saint Mary\'s':7,'Vanderbilt':10,
    'Purdue':4,'High Point':13,'Texas Tech':3,'UNC Wilmington':14,
    'St. John\'s':2,'Nebraska Omaha':15,
    'San Diego St.':11,'Xavier':11,'UCLA':7,'Utah St.':10,
    'Arkansas':10,'Kansas':7,'New Mexico':10,'Marquette':7,
    'Colorado St.':12,'Memphis':5,
    'American':16,'Saint Francis':16,
}

# Build lookup for 2025 teams
kp_lookup = kp2025.set_index('Team').to_dict('index')

def get_team_stats_2025(team_name, seed):
    # Fuzzy match
    stats = kp_lookup.get(team_name, {})
    if not stats:
        # Try partial match
        for k in kp_lookup:
            if team_name.lower() in k.lower() or k.lower() in team_name.lower():
                stats = kp_lookup[k]
                break
    
    # Construct in same format as training
    return {
        'Net Rtg (AdjEM)': stats.get('NetRtg', 0),
        'Adj ORtg': stats.get('ORtg', 105),
        'Adj DRtg': stats.get('DRtg', 105),
        'Adj Tempo': stats.get('Tempo', 68),
        'Luck': 0,
        'Pomeroy Rank': stats.get('Pom_Rank', 100),
        'Reg Wins': stats.get('Reg_Wins', 20),
        'Reg Losses': stats.get('Reg_Losses', 10),
        'Seed': seed,
    }

def predict_game(t1_name, t1_seed, t2_name, t2_seed, rnd):
    t1_stats = get_team_stats_2025(t1_name, t1_seed)
    t2_stats = get_team_stats_2025(t2_name, t2_seed)
    feats = make_matchup_features(t1_stats, t2_stats)
    feats['round'] = rnd
    fv = pd.DataFrame([feats])[feature_cols].fillna(0)
    prob = ensemble.predict_proba(fv)[0][1]  # prob t1 wins
    return prob

#Simulate the 2025 tournament with results

round_num_map = {'R64':1,'R32':2,'S16':3,'E8':4,'F4':5,'NCG':6}

predictions = []
correct = 0
total = 0

print("\n=== 2025 Tournament Predictions vs Actuals ===\n")
print(f"{'Round':<6} {'Team 1':<22} {'Team 2':<22} {'Pred Winner':<22} {'Actual Winner':<22} {'Correct'}")
print("-"*110)

for rec in results_2025_raw:
    t1_name, t1_seed, t2_name, t2_seed, t1_won, rnd_label = rec
    rnd = round_num_map[rnd_label]
    
    prob_t1 = predict_game(t1_name, t1_seed, t2_name, t2_seed, rnd)
    pred_winner = t1_name if prob_t1 >= 0.5 else t2_name
    actual_winner = t1_name if t1_won == 1 else t2_name
    is_correct = pred_winner == actual_winner
    
    if is_correct:
        correct += 1
    total += 1
    
    predictions.append({
        'Round': rnd_label,
        'Team1': t1_name, 'Seed1': t1_seed,
        'Team2': t2_name, 'Seed2': t2_seed,
        'Prob_T1_Win': round(prob_t1, 3),
        'Predicted_Winner': pred_winner,
        'Actual_Winner': actual_winner,
        'Correct': is_correct
    })
    
    mark = '✓' if is_correct else '✗'
    print(f"{rnd_label:<6} ({t1_seed}){t1_name:<19} ({t2_seed}){t2_name:<19} {pred_winner:<22} {actual_winner:<22} {mark}")

print(f"\n{'='*110}")
print(f"OVERALL: {correct}/{total} = {correct/total*100:.1f}%")

# By round
preds_df = pd.DataFrame(predictions)
print("\nBy Round:")
for rnd in ['R64','R32','S16','E8','F4','NCG']:
    rdf = preds_df[preds_df['Round']==rnd]
    if len(rdf) > 0:
        c = rdf['Correct'].sum()
        t = len(rdf)
        print(f"  {rnd}: {c}/{t} = {c/t*100:.0f}%")

#Output the results and model feature importances
preds_df.to_csv('/home/claude/predictions_2025.csv', index=False)
print("\nSaved predictions to /home/claude/predictions_2025.csv")

# Feature importance
gbm_model = ensemble.estimators_[0]
importances = pd.Series(gbm_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 Features:")
print(importances.head(10))


