import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv('CRM/Cohort_Analysis/Year 2010-2011.csv')

# Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.
df.dropna(inplace=True)

# Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.
df = df[~df['Invoice'].str.contains('C', na=False)]

df = df[(df['Quantity'] > 0)]
df = df[(df['Price'] > 0)]

# Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz.
df['TotalPrice'] = df['Quantity'] * df['Price']

df["InvoiceDate"].max()

today_date = dt.datetime(2010, 12, 11)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')

# Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm.columns = ['recency', 'frequency', 'monetary']

# Not 2: rfm dataframe’ini oluşturduktan sonra veri setini "monetary>0" olacak şekilde filtreleyiniz.
rfm = rfm[rfm['monetary'] > 0]


# Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Oluşan 2 farklı değişkenin değerini tek bir değişken olarak ifade ediniz ve RFM_SCORE olarak kaydediniz.
rfm['RFM_SCORE'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)


# Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlamaları yapınız.
# seg_map yardımı ile skorları segmentlere çeviriniz.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm = rfm[["recency", "frequency", "monetary", "segment"]]
