import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#from underthesea import word_tokenize, pos_tag, sent_tokenize
from gensim import corpora, models, similarities
import re
import matplotlib.pyplot as plt

# 1. Read data
df = pd.read_csv('data_clean.csv')

# GUI
st.title("Data Science Project")
st.write("Product Recommendation")


# 2. Data pre-processing
STOP_WORD_FILE = 'vietnamese-stopwords.txt'

with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

#st.dataframe(df.head())

# 3. Build model

products_gem = [[text for text in x.split()] for x in df.name_wt]
products_gem_re = [[t for t in text if not t in stop_words] for text in products_gem] 

dictionary = corpora.Dictionary(products_gem_re)

corpus  = [dictionary.doc2bow(text) for text in products_gem_re]

feature_cnt = len(dictionary.token2id)
# Use TF-IDF Model to process corpus, obtaining index
tfidf = models.TfidfModel(corpus)
# tính toán sự tương tự trong ma trận thưa thớt
index = similarities.SparseMatrixSimilarity(tfidf[corpus],
                                            num_features = feature_cnt)

def gensim(product_selection):
    view_product = product_selection.lower()
    view_product = re.sub('\n', ' ', view_product)
    view_product = re.sub('thông tin chi tiết', ' ', view_product)
    view_product = re.sub('mô tả sản phẩm', ' ', view_product)
    view_product = re.sub(r'[^a-zA-Z0-9\sÀ-ỹ]', ' ', view_product)  # Remove special characters except 'TV'
    view_product = re.sub(r'\d', '', view_product)  # Remove digits
    view_product = view_product.replace('  ', ' ')
    view_product = view_product.replace('  ', ' ')
    
    #lọc các từ đứng 1 mình ví dụ: h, p, s, i, v,....
    view_product = view_product.split()
    view_product = [word for word in view_product if len(view_product) > 1]

    kw_vector = dictionary.doc2bow(view_product)

    sim = index[tfidf[kw_vector]]
    sorted_indices = sim.argsort()[::-1]
    sorted_recommend = sorted_indices[1]
    df_recommend = df.iloc[sorted_recommend]
    print(view_product)
    return df_recommend['name']

#product_selection = df['text'][0]

#st.write(f"Lựa chọn của bạn là: {product_selection}\nGợi ý: {gensim(product_selection)}")


# Recommend for product
def recommender(view_product, dictionary, tfidf, index, top_n):
      view_product = view_product.lower().split()
      kw_vector = dictionary.doc2bow(view_product)
      print("View product's vector:")
      #print(kw_vector)
      # Similarity calculation
      sim = index[tfidf[kw_vector]]

      #print result
      list_id = []
      list_score = []
      for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])

      df_result = pd.DataFrame({'id': list_id, 'score': list_score})

      #five highest scores
      highest_score = df_result.sort_values(by = 'score', ascending = False).head(top_n)
      print(str(top_n) + ' highest scores:')
      print(highest_score)
      print("Ids to list:")
      idToList = list(highest_score['id'])
      print(idToList)

      products_find = df[df.index.isin(idToList)]
      results = products_find[['item_id', 'name']]
      results = pd.concat([results, highest_score], axis = 1).sort_values(by='score', ascending = False)
      return results

# sản phẩm đang xem
name_description_pre = df['name_wt'].to_string(index=False)

results = recommender(name_description_pre, dictionary, tfidf, index, 8)

def get_item_name(item_id):
  return df[df['item_id'] == item_id]['name'].iloc[0]

# Recommend for item_id
def recommend_by_id(item_id, top_n):

     print("Recommending " + str(top_n) + " products similiar to " + "item_id " + str(item_id) + ' - ' + get_item_name(item_id) + "...")
     print('* '*40)
     recommended_items_df = recommender(get_item_name(item_id), dictionary, tfidf, index, top_n)
     print(recommended_items_df)

#item_id = 48102821
#top_n = 7
#recommend_by_id(item_id, top_n)
#st.write(f"Lựa chọn của bạn là: {item_id}\nGợi ý: {recommend_by_id(item_id, top_n)}")

#recommend_by_id(48102821, 5)

# 6. GUI
menu = ["Business Objective", "Build Project", "Recommend by content"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Build a recommendation System on tiki.vn when user select a product_id and number of recommended products, the sytem will recommend accordingly based on the machine learning algorithm.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for product recommendation on tiki.vn website.""")
    st.image("Recommender_by_content.png")
    st.image("Collaborative_filtering.png")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(df.head(5))
    st.write("##### 2. Visualize product information")
    st.image("word_cloud.png")

    st.write("##### 3. Build model...")


elif choice == 'Recommend':
    # Define the Streamlit app
    #def main():
        #st.title("Product Recommendation App")

    # User input for product ID
    #item_id = st.number_input("Enter Product ID:", step = 1)


    # User input for product ID
    view_product = st.text_input("Enter some text 👇")
   
    # User input for the number of recommendations
    top_n = st.number_input("Number of Recommendations:", min_value=1, max_value=10, value=5)

    #if st.button("Recommend"):
        #if item_id:
            # Call the recommend function
            #st.header(f"Recommendations for Product {item_id}")
            #st.subheader("Top Recommendations:")
            #recommended_items_df = recommend_by_id(item_id, top_n)
            #st.dataframe(recommended_items_df)

    if st.button("Recommend"):
        if view_product:
            # Call the recommend function
            #st.header(f"Recommendations for Product_ID {item_id}") 
            #st.header(f"Recommendations for Product_ID {item_id}")          
            #view_product = df[df['item_id'] == item_id]['name'].iloc[0]
            #st.header(f"- Product_Name: {view_product}")     
            view_product = view_product.lower().split()
            kw_vector = dictionary.doc2bow(view_product)
            print("View product's vector:")
            #print(kw_vector)
            # Similarity calculation
            sim = index[tfidf[kw_vector]]

            #print result
            list_id = []
            list_score = []
            for i in range(len(sim)):
                list_id.append(i)
                list_score.append(sim[i])

            df_result = pd.DataFrame({'id': list_id, 'score': list_score})
            st.subheader(f"- Top {top_n} Recommendations:")
            #five highest scores
            highest_score = df_result.sort_values(by = 'score', ascending = False).head(top_n)
            print(str(top_n) + ' highest scores:')
            print(highest_score)
            print("Ids to list:")
            idToList = list(highest_score['id'])
            print(idToList)

            products_find = df[df.index.isin(idToList)]
            results = products_find[['item_id', 'name']]
            results = pd.concat([results, highest_score], axis = 1).sort_values(by='score', ascending = False)
            results = results[['name', 'score']]
            st.dataframe(results)