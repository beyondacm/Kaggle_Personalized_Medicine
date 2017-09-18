
## Solutions for the Kaggle: PMRCT
- Personalized Medicine : Redefining Cancer Treatment 
- Predict the effect of Genetic Variants to enable Personalized Medicine

## Loading Library


```python
import numpy as np  # Linear Algebra
import pandas as pd # Data processing
from pandas import HDFStore
import matplotlib.pyplot as plt # Data Visualization
%matplotlib inline
import seaborn as sns # Visualization
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import re
import nltk
```

## Data Importing & Preprocessing


```python
train_text_df = pd.read_csv('data/training_text', sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df  = pd.read_csv('data/test_text', sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
train_vari_df = pd.read_csv('data/training_variants')
test_vari_df  = pd.read_csv('data/test_variants')
```


```python
train_text_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Cyclin-dependent kinases (CDKs) regulate a var...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Abstract Background  Non-small cell lung canc...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Abstract Background  Non-small cell lung canc...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Recent evidence has demonstrated that acquired...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Oncogenic mutations in the monomeric Casitas B...</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_vari_df.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Gene</th>
      <th>Variation</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3316</th>
      <td>3316</td>
      <td>RUNX1</td>
      <td>D171N</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3317</th>
      <td>3317</td>
      <td>RUNX1</td>
      <td>A122*</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3318</th>
      <td>3318</td>
      <td>RUNX1</td>
      <td>Fusions</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3319</th>
      <td>3319</td>
      <td>RUNX1</td>
      <td>R80C</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3320</th>
      <td>3320</td>
      <td>RUNX1</td>
      <td>K83E</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = train_vari_df.join(train_text_df.set_index('ID'), on='ID')
```


```python
df_y = df['Class'].values
df_X = df[['Gene', 'Variation', 'Text']]
```


```python
print( type(df_y) )
print( type(df_X) )
```

    <class 'numpy.ndarray'>
    <class 'pandas.core.frame.DataFrame'>



```python
df_test = test_vari_df.join(test_text_df.set_index('ID'), on='ID')
df_test = df_test.iloc[:,1:]
df_new = pd.concat([df_X, df_test], ignore_index=True)
```


```python
print( type(df_new), df_new.shape )
```

    <class 'pandas.core.frame.DataFrame'> (8989, 3)



```python
df_new.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8989 entries, 0 to 8988
    Data columns (total 3 columns):
    Gene         8989 non-null object
    Variation    8989 non-null object
    Text         8989 non-null object
    dtypes: object(3)
    memory usage: 210.8+ KB


## Extracting Features

### TF-IDF Features


```python
tfidf_vect = TfidfVectorizer()
stop_words = ENGLISH_STOP_WORDS
```


```python
def text_Decomposition(text):
    text = re.sub(r"[^a-zA-Z0-9^,!./\+-_=]", " ", text)
    text = text.lower().split()
    text = [i for i in text if not i in stop_words]
    text = " ".join(text)
    text = text.replace("."," ").replace(","," ")
    return (text)
```


```python
df_new['Text'] = df_new['Text'].apply(text_Decomposition)
```


```python
df_new['Text']
```




    0       cyclin-dependent kinases cdks regulate variety...
    1       abstract background non-small cell lung cancer...
    2       abstract background non-small cell lung cancer...
    3       recent evidence demonstrated acquired uniparen...
    4       oncogenic mutations monomeric casitas b-lineag...
    5       oncogenic mutations monomeric casitas b-lineag...
    6       oncogenic mutations monomeric casitas b-lineag...
    7       cbl negative regulator activated receptor tyro...
    8       abstract juvenile myelomonocytic leukemia jmml...
    9       abstract juvenile myelomonocytic leukemia jmml...
    10      oncogenic mutations monomeric casitas b-lineag...
    11      noonan syndrome autosomal dominant congenital ...
    12      noonan syndrome autosomal dominant congenital ...
    13      noonan syndrome autosomal dominant congenital ...
    14      oncogenic mutations monomeric casitas b-lineag...
    15      noonan syndrome autosomal dominant congenital ...
    16      determine residual cylindrical refractive erro...
    17      acquired uniparental disomy aupd common featur...
    18      oncogenic mutations monomeric casitas b-lineag...
    19      acquired uniparental disomy aupd common featur...
    20      abstract background non-small cell lung cancer...
    21      oncogenic mutations monomeric casitas b-lineag...
    22      oncogenic mutations monomeric casitas b-lineag...
    23      recent evidence demonstrated acquired uniparen...
    24      recent evidence demonstrated acquired uniparen...
    25      recent evidence demonstrated acquired uniparen...
    26      abstract n-myristoylation common form co-trans...
    27      heterozygous mutations telomerase components t...
    28      sequencing studies identified recurrent coding...
    29      heterozygous mutations telomerase components t...
                                  ...                        
    8959    using dna microarray approach screen gene copy...
    8960    tumor suppressor protein p53 inactivated mutat...
    8961    mutational analysis oncogenes critical underst...
    8962    serine/threonine protein kinase encoded akf pr...
    8963    common participation oncogenic kras proteins l...
    8964    ezh2 enhancer zeste homolog 2 critical enzymat...
    8965    mutations metabolic enzymes isocitrate dehydro...
    8966    pancreatic carcinomas acinar differentiation  ...
    8967    -catenin-mediated signaling constitutively act...
    8968    summary genetic abnormalities underlying hered...
    8969    lung cancer leading cause cancer-related morta...
    8970    summary past decade  treatment lung adenocarci...
    8971    transcription factor tumor suppressor protein ...
    8972    protein tyrosine phosphatase receptor type d p...
    8973    sensitizing activating mutations tyrosine kina...
    8974    structural rearrangements chromosome 10 freque...
    8975    introduction production fertile gametes essent...
    8976    checkpoint kinase 2 chek2  chk2 emerges import...
    8977    introduction telomere sequences chromosomal en...
    8978    estimated 1 million cases breast cancer bc dia...
    8979    occurring responders expression table i359l 0 ...
    8980    background aims: inherited deleterious mutatio...
    8981    glioblastoma multiforme gbm lethal brain tumou...
    8982    diffuse large b cell lymphoma dlbcl complex di...
    8983    figure largedownload s818l clones change atpas...
    8984    realization late 1970s ras harboured transform...
    8985    hemizygous deletions common molecular abnormal...
    8986    r267w smartpool investigate 533 experiments 5q...
    8987    abstract blood samples 125 unrelated families ...
    8988    loss dna mismatch repair mmr humans  mainly mu...
    Name: Text, Length: 8989, dtype: object




```python
tfidf_features = tfidf_vect.fit_transform(df_new['Text'])
```


```python
print( type(tfidf_features), tfidf_features.get_shape() )
```

    <class 'scipy.sparse.csr.csr_matrix'> (8989, 167304)



```python
svd = TruncatedSVD(n_components=500, n_iter=5, random_state=0)
```


```python
truncated_tfidf = svd.fit_transform(tfidf_features)
```


```python
print( type( truncated_tfidf ), truncated_tfidf.shape )
```

    <class 'numpy.ndarray'> (8989, 500)



```python
df_tfidf_col_name = ["tfidf_"+str(i) for i in range(500)]
```


```python
df_tfidf = pd.DataFrame( truncated_tfidf )
df_tfidf.columns = df_tfidf_col_name
df_tfidf.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tfidf_0</th>
      <th>tfidf_1</th>
      <th>tfidf_2</th>
      <th>tfidf_3</th>
      <th>tfidf_4</th>
      <th>tfidf_5</th>
      <th>tfidf_6</th>
      <th>tfidf_7</th>
      <th>tfidf_8</th>
      <th>tfidf_9</th>
      <th>...</th>
      <th>tfidf_490</th>
      <th>tfidf_491</th>
      <th>tfidf_492</th>
      <th>tfidf_493</th>
      <th>tfidf_494</th>
      <th>tfidf_495</th>
      <th>tfidf_496</th>
      <th>tfidf_497</th>
      <th>tfidf_498</th>
      <th>tfidf_499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8984</th>
      <td>0.198822</td>
      <td>-0.050257</td>
      <td>0.012171</td>
      <td>-0.089914</td>
      <td>-0.004059</td>
      <td>-0.027756</td>
      <td>0.009028</td>
      <td>-0.050974</td>
      <td>0.080609</td>
      <td>0.033432</td>
      <td>...</td>
      <td>-0.017983</td>
      <td>-0.005827</td>
      <td>0.009936</td>
      <td>-0.000408</td>
      <td>0.009475</td>
      <td>0.011130</td>
      <td>0.003676</td>
      <td>0.011604</td>
      <td>-0.019133</td>
      <td>-0.003256</td>
    </tr>
    <tr>
      <th>8985</th>
      <td>0.168936</td>
      <td>-0.042271</td>
      <td>-0.007801</td>
      <td>-0.062719</td>
      <td>0.020268</td>
      <td>-0.023329</td>
      <td>-0.006039</td>
      <td>-0.027311</td>
      <td>0.033846</td>
      <td>-0.007164</td>
      <td>...</td>
      <td>-0.013675</td>
      <td>0.004163</td>
      <td>-0.009727</td>
      <td>-0.005671</td>
      <td>-0.007209</td>
      <td>0.031979</td>
      <td>0.007391</td>
      <td>0.003575</td>
      <td>0.033547</td>
      <td>0.010747</td>
    </tr>
    <tr>
      <th>8986</th>
      <td>0.242678</td>
      <td>-0.099755</td>
      <td>-0.108291</td>
      <td>0.137711</td>
      <td>0.065636</td>
      <td>0.127120</td>
      <td>-0.129976</td>
      <td>0.030754</td>
      <td>-0.017880</td>
      <td>-0.014541</td>
      <td>...</td>
      <td>-0.017030</td>
      <td>0.006821</td>
      <td>0.006546</td>
      <td>-0.003729</td>
      <td>0.009402</td>
      <td>0.022159</td>
      <td>-0.002369</td>
      <td>-0.004654</td>
      <td>-0.022835</td>
      <td>-0.002433</td>
    </tr>
    <tr>
      <th>8987</th>
      <td>0.163984</td>
      <td>-0.025351</td>
      <td>0.004394</td>
      <td>-0.007817</td>
      <td>0.021132</td>
      <td>-0.008803</td>
      <td>-0.003844</td>
      <td>-0.003872</td>
      <td>-0.000813</td>
      <td>-0.000259</td>
      <td>...</td>
      <td>-0.008726</td>
      <td>-0.007982</td>
      <td>-0.004096</td>
      <td>-0.004657</td>
      <td>-0.005381</td>
      <td>0.010965</td>
      <td>0.006184</td>
      <td>0.001998</td>
      <td>-0.004294</td>
      <td>-0.003493</td>
    </tr>
    <tr>
      <th>8988</th>
      <td>0.169593</td>
      <td>0.042018</td>
      <td>0.020699</td>
      <td>-0.015233</td>
      <td>0.007866</td>
      <td>-0.033753</td>
      <td>-0.021334</td>
      <td>0.048231</td>
      <td>-0.030702</td>
      <td>-0.012852</td>
      <td>...</td>
      <td>-0.000946</td>
      <td>0.009195</td>
      <td>0.008646</td>
      <td>-0.001500</td>
      <td>0.003134</td>
      <td>-0.014157</td>
      <td>0.007950</td>
      <td>-0.001908</td>
      <td>0.012477</td>
      <td>-0.014073</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 500 columns</p>
</div>



### Bag of words Features


```python
bow_vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
bow_features = bow_vectorizer.fit_transform(df_new['Text'])
```


```python
print(type(bow_features), bow_features.get_shape())
```

    <class 'scipy.sparse.csr.csr_matrix'> (8989, 167304)



```python
svd_bow = TruncatedSVD(n_components=500, n_iter=5, random_state=0)
truncated_bow = svd_bow.fit_transform( bow_features )
```


```python
print( type( truncated_bow ), truncated_bow.shape )
```

    <class 'numpy.ndarray'> (8989, 500)



```python
df_bow_col_name = ["bow_"+str(i) for i in range(500)]
df_bow = pd.DataFrame( truncated_bow )
df_bow.columns = df_bow_col_name
df_bow.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bow_0</th>
      <th>bow_1</th>
      <th>bow_2</th>
      <th>bow_3</th>
      <th>bow_4</th>
      <th>bow_5</th>
      <th>bow_6</th>
      <th>bow_7</th>
      <th>bow_8</th>
      <th>bow_9</th>
      <th>...</th>
      <th>bow_490</th>
      <th>bow_491</th>
      <th>bow_492</th>
      <th>bow_493</th>
      <th>bow_494</th>
      <th>bow_495</th>
      <th>bow_496</th>
      <th>bow_497</th>
      <th>bow_498</th>
      <th>bow_499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8984</th>
      <td>194.526113</td>
      <td>-47.991367</td>
      <td>-23.033461</td>
      <td>-44.469582</td>
      <td>12.314702</td>
      <td>112.890376</td>
      <td>-6.968366</td>
      <td>-39.409591</td>
      <td>35.309063</td>
      <td>48.143541</td>
      <td>...</td>
      <td>6.041960</td>
      <td>-1.352358</td>
      <td>7.512079</td>
      <td>0.280627</td>
      <td>-1.234230</td>
      <td>-2.138932</td>
      <td>2.974048</td>
      <td>2.003135</td>
      <td>-2.488450</td>
      <td>-3.096574</td>
    </tr>
    <tr>
      <th>8985</th>
      <td>69.715744</td>
      <td>-15.101818</td>
      <td>-9.751791</td>
      <td>6.825362</td>
      <td>4.400990</td>
      <td>8.165505</td>
      <td>-8.844331</td>
      <td>-5.926672</td>
      <td>14.725673</td>
      <td>15.641581</td>
      <td>...</td>
      <td>1.858497</td>
      <td>2.571454</td>
      <td>0.303478</td>
      <td>-3.180141</td>
      <td>0.454055</td>
      <td>-1.137666</td>
      <td>0.840475</td>
      <td>0.636456</td>
      <td>1.383821</td>
      <td>-0.293513</td>
    </tr>
    <tr>
      <th>8986</th>
      <td>57.750708</td>
      <td>48.205872</td>
      <td>-13.017993</td>
      <td>-8.674400</td>
      <td>20.067003</td>
      <td>-23.089222</td>
      <td>-9.090631</td>
      <td>-15.952598</td>
      <td>-1.722238</td>
      <td>18.967338</td>
      <td>...</td>
      <td>2.248561</td>
      <td>0.538540</td>
      <td>1.048798</td>
      <td>0.916919</td>
      <td>-1.144835</td>
      <td>0.955626</td>
      <td>-0.159801</td>
      <td>1.170221</td>
      <td>2.212243</td>
      <td>-1.639418</td>
    </tr>
    <tr>
      <th>8987</th>
      <td>210.185671</td>
      <td>7.402431</td>
      <td>9.203735</td>
      <td>-33.907666</td>
      <td>-30.323195</td>
      <td>-12.265160</td>
      <td>-18.975167</td>
      <td>-24.648927</td>
      <td>48.958101</td>
      <td>-0.342035</td>
      <td>...</td>
      <td>-5.108713</td>
      <td>5.970898</td>
      <td>2.408793</td>
      <td>2.173039</td>
      <td>1.585976</td>
      <td>11.776157</td>
      <td>-3.472449</td>
      <td>4.134649</td>
      <td>2.737938</td>
      <td>-5.758787</td>
    </tr>
    <tr>
      <th>8988</th>
      <td>69.524366</td>
      <td>-2.637226</td>
      <td>34.608404</td>
      <td>-0.423025</td>
      <td>-0.272547</td>
      <td>-2.477183</td>
      <td>-1.825543</td>
      <td>2.157657</td>
      <td>16.846462</td>
      <td>5.286870</td>
      <td>...</td>
      <td>0.779397</td>
      <td>-0.458060</td>
      <td>-0.016754</td>
      <td>-2.339020</td>
      <td>-1.451003</td>
      <td>2.258365</td>
      <td>4.500253</td>
      <td>0.275865</td>
      <td>0.884756</td>
      <td>0.543471</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 500 columns</p>
</div>



### Dummy Features


```python
df_dummy = df_new.iloc[:,:2]
```


```python
df_dummy
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gene</th>
      <th>Variation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FAM58A</td>
      <td>Truncating Mutations</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBL</td>
      <td>W802*</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CBL</td>
      <td>Q249E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CBL</td>
      <td>N454D</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CBL</td>
      <td>L399V</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CBL</td>
      <td>V391I</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CBL</td>
      <td>V430M</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CBL</td>
      <td>Deletion</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CBL</td>
      <td>Y371H</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CBL</td>
      <td>C384R</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CBL</td>
      <td>P395A</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CBL</td>
      <td>K382E</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CBL</td>
      <td>R420Q</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CBL</td>
      <td>C381A</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CBL</td>
      <td>P428L</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CBL</td>
      <td>D390Y</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CBL</td>
      <td>Truncating Mutations</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CBL</td>
      <td>Q367P</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CBL</td>
      <td>M374V</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CBL</td>
      <td>Y371S</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CBL</td>
      <td>H94Y</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CBL</td>
      <td>C396R</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CBL</td>
      <td>G375P</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CBL</td>
      <td>S376F</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CBL</td>
      <td>P417A</td>
    </tr>
    <tr>
      <th>25</th>
      <td>CBL</td>
      <td>H398Y</td>
    </tr>
    <tr>
      <th>26</th>
      <td>SHOC2</td>
      <td>S2G</td>
    </tr>
    <tr>
      <th>27</th>
      <td>TERT</td>
      <td>Y846C</td>
    </tr>
    <tr>
      <th>28</th>
      <td>TERT</td>
      <td>C228T</td>
    </tr>
    <tr>
      <th>29</th>
      <td>TERT</td>
      <td>H412Y</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8959</th>
      <td>VSX1</td>
      <td>R166W</td>
    </tr>
    <tr>
      <th>8960</th>
      <td>MTM1</td>
      <td>E157K</td>
    </tr>
    <tr>
      <th>8961</th>
      <td>D2HGDH</td>
      <td>V444A</td>
    </tr>
    <tr>
      <th>8962</th>
      <td>DMD</td>
      <td>Y231N</td>
    </tr>
    <tr>
      <th>8963</th>
      <td>ANKH</td>
      <td>G389R</td>
    </tr>
    <tr>
      <th>8964</th>
      <td>DCX</td>
      <td>R59L</td>
    </tr>
    <tr>
      <th>8965</th>
      <td>ADSL</td>
      <td>R190Q</td>
    </tr>
    <tr>
      <th>8966</th>
      <td>HSD17B3</td>
      <td>A56T</td>
    </tr>
    <tr>
      <th>8967</th>
      <td>CD96</td>
      <td>T280M</td>
    </tr>
    <tr>
      <th>8968</th>
      <td>HPS3</td>
      <td>R397W</td>
    </tr>
    <tr>
      <th>8969</th>
      <td>FKTN</td>
      <td>R179T</td>
    </tr>
    <tr>
      <th>8970</th>
      <td>DARS2</td>
      <td>L613F</td>
    </tr>
    <tr>
      <th>8971</th>
      <td>TP53</td>
      <td>G245C</td>
    </tr>
    <tr>
      <th>8972</th>
      <td>ALG12</td>
      <td>T67M</td>
    </tr>
    <tr>
      <th>8973</th>
      <td>ACOX1</td>
      <td>Q309R</td>
    </tr>
    <tr>
      <th>8974</th>
      <td>CLDN19</td>
      <td>Q57E</td>
    </tr>
    <tr>
      <th>8975</th>
      <td>MLH3</td>
      <td>N499S</td>
    </tr>
    <tr>
      <th>8976</th>
      <td>GJB1</td>
      <td>F235C</td>
    </tr>
    <tr>
      <th>8977</th>
      <td>LRP5</td>
      <td>G171V</td>
    </tr>
    <tr>
      <th>8978</th>
      <td>TGFBI</td>
      <td>R124S</td>
    </tr>
    <tr>
      <th>8979</th>
      <td>CYP2C9</td>
      <td>I359L</td>
    </tr>
    <tr>
      <th>8980</th>
      <td>HSD17B3</td>
      <td>M235V</td>
    </tr>
    <tr>
      <th>8981</th>
      <td>CAV3</td>
      <td>A46V</td>
    </tr>
    <tr>
      <th>8982</th>
      <td>ABHD5</td>
      <td>E260K</td>
    </tr>
    <tr>
      <th>8983</th>
      <td>NR3C2</td>
      <td>S818L</td>
    </tr>
    <tr>
      <th>8984</th>
      <td>SLC46A1</td>
      <td>R113S</td>
    </tr>
    <tr>
      <th>8985</th>
      <td>FOXC1</td>
      <td>L130F</td>
    </tr>
    <tr>
      <th>8986</th>
      <td>GSS</td>
      <td>R267W</td>
    </tr>
    <tr>
      <th>8987</th>
      <td>CTSK</td>
      <td>G79E</td>
    </tr>
    <tr>
      <th>8988</th>
      <td>DFNB59</td>
      <td>T54I</td>
    </tr>
  </tbody>
</table>
<p>8989 rows × 2 columns</p>
</div>




```python
df_dummy = pd.get_dummies(df_dummy)
```


```python
df_dummy.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gene_A4GALT</th>
      <th>Gene_AAAS</th>
      <th>Gene_AANAT</th>
      <th>Gene_AARS</th>
      <th>Gene_ABCA1</th>
      <th>Gene_ABCA12</th>
      <th>Gene_ABCA3</th>
      <th>Gene_ABCA4</th>
      <th>Gene_ABCB11</th>
      <th>Gene_ABCB7</th>
      <th>...</th>
      <th>Variation_null380R</th>
      <th>Variation_null399R</th>
      <th>Variation_null420W</th>
      <th>Variation_null423L</th>
      <th>Variation_null462G</th>
      <th>Variation_null483L</th>
      <th>Variation_null496R</th>
      <th>Variation_null522S</th>
      <th>Variation_null654G</th>
      <th>Variation_p61BRAF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8984</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8985</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8986</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8987</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8988</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 10116 columns</p>
</div>



### Combine TF-IDF Features + Bow Features + Dummy Features


```python
df_dummy['tmp'] = [i for i in range(len(df_dummy))]
df_bow['tmp']   = [i for i in range(len(df_bow))]
df_tfidf['tmp'] = [i for i in range(len(df_tfidf))]

df_new = df_bow.join(df_tfidf.set_index("tmp"), on="tmp")
df_new = df_new.join(df_dummy.set_index("tmp"), on="tmp")
del df_new['tmp']
```


```python
print( type(df_new), df_new.shape )
```

    <class 'pandas.core.frame.DataFrame'> (8989, 11116)



```python
df_new.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gene_A4GALT</th>
      <th>Gene_AAAS</th>
      <th>Gene_AANAT</th>
      <th>Gene_AARS</th>
      <th>Gene_ABCA1</th>
      <th>Gene_ABCA12</th>
      <th>Gene_ABCA3</th>
      <th>Gene_ABCA4</th>
      <th>Gene_ABCB11</th>
      <th>Gene_ABCB7</th>
      <th>...</th>
      <th>bow_490</th>
      <th>bow_491</th>
      <th>bow_492</th>
      <th>bow_493</th>
      <th>bow_494</th>
      <th>bow_495</th>
      <th>bow_496</th>
      <th>bow_497</th>
      <th>bow_498</th>
      <th>bow_499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8984</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6.041960</td>
      <td>-1.352358</td>
      <td>7.512079</td>
      <td>0.280627</td>
      <td>-1.234230</td>
      <td>-2.138932</td>
      <td>2.974048</td>
      <td>2.003135</td>
      <td>-2.488450</td>
      <td>-3.096574</td>
    </tr>
    <tr>
      <th>8985</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.858497</td>
      <td>2.571454</td>
      <td>0.303478</td>
      <td>-3.180141</td>
      <td>0.454055</td>
      <td>-1.137666</td>
      <td>0.840475</td>
      <td>0.636456</td>
      <td>1.383821</td>
      <td>-0.293513</td>
    </tr>
    <tr>
      <th>8986</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2.248561</td>
      <td>0.538540</td>
      <td>1.048798</td>
      <td>0.916919</td>
      <td>-1.144835</td>
      <td>0.955626</td>
      <td>-0.159801</td>
      <td>1.170221</td>
      <td>2.212243</td>
      <td>-1.639418</td>
    </tr>
    <tr>
      <th>8987</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>-5.108713</td>
      <td>5.970898</td>
      <td>2.408793</td>
      <td>2.173039</td>
      <td>1.585976</td>
      <td>11.776157</td>
      <td>-3.472449</td>
      <td>4.134649</td>
      <td>2.737938</td>
      <td>-5.758787</td>
    </tr>
    <tr>
      <th>8988</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.779397</td>
      <td>-0.458060</td>
      <td>-0.016754</td>
      <td>-2.339020</td>
      <td>-1.451003</td>
      <td>2.258365</td>
      <td>4.500253</td>
      <td>0.275865</td>
      <td>0.884756</td>
      <td>0.543471</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 11116 columns</p>
</div>




```python
df_X     = df_new.iloc[:3321,  :]
df_test  = df_new.iloc[3321:,  :]
```


```python
df_X.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gene_A4GALT</th>
      <th>Gene_AAAS</th>
      <th>Gene_AANAT</th>
      <th>Gene_AARS</th>
      <th>Gene_ABCA1</th>
      <th>Gene_ABCA12</th>
      <th>Gene_ABCA3</th>
      <th>Gene_ABCA4</th>
      <th>Gene_ABCB11</th>
      <th>Gene_ABCB7</th>
      <th>...</th>
      <th>bow_490</th>
      <th>bow_491</th>
      <th>bow_492</th>
      <th>bow_493</th>
      <th>bow_494</th>
      <th>bow_495</th>
      <th>bow_496</th>
      <th>bow_497</th>
      <th>bow_498</th>
      <th>bow_499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3316</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1.745059</td>
      <td>-5.821573</td>
      <td>3.241390</td>
      <td>-4.601446</td>
      <td>-0.799721</td>
      <td>0.964183</td>
      <td>-6.654517</td>
      <td>-7.173837</td>
      <td>-0.191026</td>
      <td>-3.266606</td>
    </tr>
    <tr>
      <th>3317</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2.279242</td>
      <td>-2.976092</td>
      <td>1.965532</td>
      <td>-8.854997</td>
      <td>3.721709</td>
      <td>2.073133</td>
      <td>-9.029381</td>
      <td>-10.325801</td>
      <td>-7.744560</td>
      <td>-3.181727</td>
    </tr>
    <tr>
      <th>3318</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>-2.660800</td>
      <td>-1.165723</td>
      <td>6.478892</td>
      <td>0.090645</td>
      <td>5.618230</td>
      <td>-3.316006</td>
      <td>-4.345822</td>
      <td>-6.997560</td>
      <td>1.835467</td>
      <td>-5.563675</td>
    </tr>
    <tr>
      <th>3319</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>-3.003526</td>
      <td>2.542012</td>
      <td>-2.760956</td>
      <td>0.071038</td>
      <td>-1.042046</td>
      <td>-1.497018</td>
      <td>2.630176</td>
      <td>0.224616</td>
      <td>6.731985</td>
      <td>-0.477152</td>
    </tr>
    <tr>
      <th>3320</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>-2.288296</td>
      <td>3.421923</td>
      <td>-2.794622</td>
      <td>-1.133969</td>
      <td>-4.935845</td>
      <td>-4.073437</td>
      <td>4.769763</td>
      <td>1.425812</td>
      <td>9.841368</td>
      <td>-6.163921</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 11116 columns</p>
</div>




```python
train_X, test_X, train_y, test_y = train_test_split(df_X, df_y, random_state=0)
```

## Model Building


```python
# Cross Validation
def model_cv(train, test, train_y, test_y, model, name):
    model.fit(train, train_y)
    print(name,': ',model.best_params_)
    pred_y = model.predict_proba(test)
    print('train score: {}'.format(model.score(train, train_y)))
    print('test score: {}'.format(model.score(test, test_y)))
    print('log loss: {}'.format(log_loss(test_y, pred_y)))
    print()
```


```python
# Models
def forest(train, test, train_y, test_y):
    param = [{'n_estimators':[500],
              'max_features': ['sqrt']
         }]
    model = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=0), param, cv=StratifiedKFold(random_state=0))
    name = 'Random forest'
    return model_cv(train, test, train_y, test_y, model, name)

def xgbc(train, test, train_y, test_y):
    param = [{'n_estimators': [300],
         'learning_rate': [0.05],}]
    model = GridSearchCV(XGBClassifier(), param, cv=StratifiedKFold(random_state=0))
    name = 'XGBoost'
    return model_cv(train, test, train_y, test_y, model, name)

def lgbm(train, test, train_y, test_y):
    param = [{'n_estimators': [100],
         'learning_rate': [0.05]}]
    model = GridSearchCV(LGBMClassifier(), param, cv=3)
    name = 'LightGBM'
    return model_cv(train, test, train_y, test_y, model, name)
```


```python
forest(train_X, test_X, train_y, test_y)
```

    Random forest :  {'max_features': 'sqrt', 'n_estimators': 500}
    train score: 1.0
    test score: 0.6293622141997594
    log loss: 1.4546850488576162
    



```python
lgbm(train_X, test_X, train_y, test_y)
```

    LightGBM :  {'learning_rate': 0.05, 'n_estimators': 100}
    train score: 0.9080321285140562
    test score: 0.6702767749699158
    log loss: 0.9912064764789812
    



```python
# xgbc(train_X, test_X, train_y, test_y)
```


```python
# we select model lgbm for our task
lgbm = LGBMClassifier(learning_rate=0.05, n_estimators=100)
```


```python
lgbm.fit(train_X, train_y)
```




    LGBMClassifier(boosting_type='gbdt', colsample_bytree=1, learning_rate=0.05,
            max_bin=255, max_depth=-1, min_child_samples=10,
            min_child_weight=5, min_split_gain=0, n_estimators=100, nthread=-1,
            num_leaves=31, objective='multiclass', reg_alpha=0, reg_lambda=0,
            seed=0, silent=True, subsample=1, subsample_for_bin=50000,
            subsample_freq=1)



## Submit


```python
pred = lgbm.predict(df_test)
pred_pro = lgbm.predict_proba(df_test)
```


```python
print( type( pred ), pred.shape, pred_pro.shape )
```

    <class 'numpy.ndarray'> (5668,) (5668, 9)



```python
pred
```




    array([7, 4, 7, ..., 2, 7, 4])




```python
from sklearn import preprocessing
```


```python
lb = preprocessing.LabelBinarizer()
lb.fit(pred)
```




    LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)




```python
lb.classes_
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
pred = lb.transform(pred)
```


```python
print( type( pred ), pred.shape )
```

    <class 'numpy.ndarray'> (5668, 9)



```python
pred = pd.DataFrame(pred)
```


```python
pred.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5663</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5664</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5665</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5666</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5667</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit = pd.DataFrame(pred)
```


```python
len(submit)
```




    5668




```python
# submit.tail()
ID = pd.DataFrame([{"ID": i} for i in range(len(submit)) ])
submit.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
```


```python
submit.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class1</th>
      <th>class2</th>
      <th>class3</th>
      <th>class4</th>
      <th>class5</th>
      <th>class6</th>
      <th>class7</th>
      <th>class8</th>
      <th>class9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5663</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5664</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5665</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5666</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5667</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ID
submit = pd.concat([ID, submit], axis=1)
submit.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>class1</th>
      <th>class2</th>
      <th>class3</th>
      <th>class4</th>
      <th>class5</th>
      <th>class6</th>
      <th>class7</th>
      <th>class8</th>
      <th>class9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5663</th>
      <td>5663</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5664</th>
      <td>5664</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5665</th>
      <td>5665</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5666</th>
      <td>5666</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5667</th>
      <td>5667</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit.to_csv('data/submit_xgbc.csv', index=False)
```


```python

```
