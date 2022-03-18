import datetime as dt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_excel('/home/kan/Workspace/DSMLBC7/Week3/Odevler/online_retail_II.xlsx', sheet_name='Year 2010-2011')

df.isnull().sum()
df.dropna(inplace=True)
df = df[~df['Invoice'].str.contains('C', na=False)]
df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]

df['TotalPrice'] = df['Quantity'] * df['Price']

rfm = df.groupby('Customer ID').agg({
                              'InvoiceDate': lambda InvoiceDate: (dt.datetime(2011, 12, 11) - InvoiceDate.max()).days,
                              'Invoice': lambda Invoice: Invoice.nunique(),
                              'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.columns = ['recency', 'frequency', 'monetary']

rfm_ = rfm.copy()
sc = MinMaxScaler((0, 1))
rfm = sc.fit_transform(rfm_)

kmeans = KMeans(n_clusters=4)
k_fit = kmeans.fit(rfm_)

k_fit.n_clusters  # küme sayısı
k_fit.cluster_centers_  # küme merkezleri
k_fit.labels_  # etiketler
k_fit.inertia_  # sse

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(rfm)
elbow.show()

elbow.elbow_value_

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(rfm)
kumeler = kmeans.labels_
pd.DataFrame({"Müşteriler": rfm_.index, "Kumeler": kumeler})
rfm_["cluster_no"] = kumeler
rfm_["cluster_no"] = rfm_["cluster_no"] + 1

rfm_.head()





