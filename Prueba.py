import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

gisette_data = pd.read_pickle(r'gisette.pickle')
training_data = gisette_data['training']['data']
training_labels = gisette_data['training']['labels']

X = pd.DataFrame(data=training_data, columns=['pixel_'+str(i) for i in range(len(training_data[0]))])
# X.info(verbose=True, show_counts = True)

# ax = plt.subplots(1,1,figsize = (10,8))
# sns.countplot(x=training_labels)
# plt.title("Labels count")
# plt.show()

# Pruebas para ver la imagen
first_image = training_data[0].reshape(200,25)
# mascara = np.logical_or(first_image == 0.0, first_image == 983.0, first_image == 991.0)
# filtered_image = np.where(mascara,first_image,0.0)
plt.imshow(first_image,cmap='gray')
plt.colorbar()
plt.show()

# valores_unicos, frecuencias = np.unique(first_image, return_counts=True)

# # Imprimir los valores únicos y sus frecuencias
# for valor, frecuencia in zip(valores_unicos, frecuencias):
#     print(f"Valor: {valor}, Frecuencia: {frecuencia}")

# valores_repetidos_100_veces_o_mas = valores_unicos[frecuencias >= 100]

# # Imprimir los valores que se repiten al menos 100 veces
# print("Valores que se repiten al menos 100 veces:")
# print(valores_repetidos_100_veces_o_mas)
# print("Cantidad de valores sin el color negro:")
# print(np.sum(first_image!=0.0))


def corr_feature_detect(data, threshold):
    corrmat = data.corr(method = 'spearman')
    corrmat = corrmat.abs().unstack() # absolute value of corr coef
    corrmat = corrmat.sort_values(ascending=False)
    corrmat = corrmat[corrmat >= threshold]
    corrmat = corrmat[corrmat < 1] # remove the digonal
    corrmat = pd.DataFrame(corrmat).reset_index()
    corrmat.columns = ['feature1', 'feature2', 'corr']
   
    grouped_feature_ls = []
    correlated_groups = []
    
    for feature in corrmat.feature1.unique():
        if feature not in grouped_feature_ls:
    
            # find all features correlated to a single feature
            correlated_block = corrmat[corrmat.feature1 == feature]
            grouped_feature_ls = grouped_feature_ls + list(
                correlated_block.feature2.unique()) + [feature]
    
            # append the block of features to the list
            correlated_groups.append(correlated_block)
    return correlated_groups

corr = corr_feature_detect(data=X,threshold=0.95)
for i in corr:
    print(i,'\n')