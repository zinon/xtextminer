from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS, Isomap
import matplotlib.pyplot as plt
from scipy.sparse.csr import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from typing import List
import seaborn as sns

from xClusters import xClusters

class xRepresentation:
    """
    Perform dimensionality reduction and represent 2d matrices 
    """
    def __init__(self, **kwargs):
        """
        input parameters
        """
        self.__n_clusters = kwargs.get('n_clusters', 2)
        self.__pca_components = kwargs.get('pca_components', 3)
        self.__tsne_components = kwargs.get('tsne_components', 3)
        self.__components = kwargs.get('components', 2)
        self.__neighbors = kwargs.get('neighbors', 2) #10
        self.__lsa_normalization = False

        #sparse matrix
        self.__matrix = None

        #tf-idf dataframe
        self.__tfidf_dataframe = pd.DataFrame()
        self.__corpora_names = None
        self.__feature_names = None

        #
        self.__clusters = xClusters(n_clusters = self.__n_clusters)
        
        #angular similarity
        self.__tfidf_angle_similarity_dataframe = pd.DataFrame()
        

        #pca
        self.__pca = PCA(n_components = self.__pca_components)

        #svd/lsa
        self.__svd = TruncatedSVD(n_components = self.__components,
                                  n_iter = 7,
                                  random_state = 33)
        
        #t-sne
        self.__tsne = TSNE(n_components = self.__components)

        #MDS
        self.__mds = MDS(n_components = self.__components,
                         random_state = 33)

        # manifold isomap for non-linear dimension reduction
        self.__isomap = Isomap(n_components=self.__components,
                               n_neighbors = self.__neighbors,
                               eigen_solver='auto')
        

        #cluster centers
        self.__kmeans_cluster_centers = None
        
        #normalizer
        self.__normalizer = Normalizer(copy=False)

    @property
    def clusters(self):
        """
        returns an xCluster class object
        """
        return self.__clusters

    @property
    def matrix(self) -> np.matrix:
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix:csr_matrix):
        if matrix.size:
            self.__matrix = matrix.todense()

    @property
    def tfidf_dataframe(self) -> pd.DataFrame():
        return self.__tfidf_dataframe

    @tfidf_dataframe.setter
    def tfidf_dataframe(self, df:pd.DataFrame() = None):
        self.__tfidf_dataframe = df

    @property
    def corpora_names(self) -> List[str]:
        return self.__corpora_names

    @corpora_names.setter
    def corpora_names(self, names = None):
        if names:
            self.__corpora_names = names

    @property
    def feature_names(self) -> List[str]:
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, names = None):
        if names:
            self.__feature_names = names

    @property
    def tfidf_angle_similarity_dataframe(self):
        return self.__tfidf_angle_similarity_dataframe
    
    @tfidf_angle_similarity_dataframe.setter
    def tfidf_angle_similarity_dataframe(self, df:pd.DataFrame() = None):
        self.__tfidf_angle_similarity_dataframe = df
                
    def __normalize_matrix(self):
         transformer = self.__normalizer.fit(self.__matrix) # fit does nothing
         self.__matrix = transformer.transform(self.__matrix)

        
    def __fit_pca(self):
        
        #Fit the model with X.
        pca = self.__pca.fit(X=self.__matrix)

        #Apply dimensionality reduction to X.
        data2d = pca.transform(X=self.__matrix)

        print("PCA data shape", data2d.shape)
                
        #calculate the cluster enters on the reduced data
        pca_cluster_centers2d = pca.transform(self.__clusters.kmeans_cluster_centers)

        n_components = self.__pca.components_.shape[0]

        print("Number of PCA components", n_components)

        #cross-check
        if n_components != data2d.shape[1]:
            print(f'inconcistent PCA companents {n_components} '
                  f'and 2d data row colunmns {data2d.shape[1]}')
            return False
        
        y = self.__clusters.kmeans_clusters_pred
        pairs = list(range(0, n_components))
        for i,j in zip(pairs, pairs[1:] + pairs[:1]):
            print("PCA components", i,j)

            #plot
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
            ax.set_title('PCA')

            #plt.scatter(data2d[:, i],
            #            data2d[:, j],
            #            c = self.__kmeans_clusters_pred) 

            for icluster in range(self.__clusters.n_actual_clusters):
                print(f"Cluster {icluster}/{self.__clusters.n_actual_clusters}")
                ax.scatter(np.array(data2d[y==icluster, i]),
                           np.array(data2d[y==icluster, j]),
                           s = 100,
                           #c = i,
                           alpha=0.5,
                           label = f"Cluster {icluster}")


            plt.scatter(pca_cluster_centers2d[:, i],
                        pca_cluster_centers2d[:, j], 
                        marker='x', s=200, linewidths=3, c='r')

            ax.set_xlabel(f"component {i}")
            ax.set_ylabel(f"component {j}")
            ax.legend(loc="best")
            plt.show()


    def __fit_lsa(self):
        #LSA/SVD results are not normalized
        #Normalization might be needed
        if self.__lsa_normalization:
            lsa = make_pipeline(self.__svd, self.__normalizer)
            lsa_data2d = lsa.fit_transform(X=self.__matrix)

        else:
            #Fit the model with X.
            lsa = self.__svd.fit(X=self.__matrix)

            #Apply dimensionality reduction to X.
            lsa_data2d = lsa.transform(X=self.__matrix)


        #calculate the cluster enters on the reduced data
        lsa_cluster_centers2d = lsa.transform(self.__kmeans_clusters.cluster_centers_)
        
        #plot
        plt.scatter(lsa_data2d[:,0],
                    lsa_data2d[:,1],
                    c = self.__kmeans_clusters_pred) #kmeans_clusters_pred, kmeans_labels

        plt.scatter(lsa_cluster_centers2d[:,0],
                    lsa_cluster_centers2d[:,1], 
                    marker='x', s=200, linewidths=3, c='r')

        plt.show()
        
    def __fit_tsne(self):
        #Fit X into an embedded space.
        #tsne = self.__tsne.fit(X=self.__matrix)

        #Fit X into an embedded space and return that transformed output.
        #Output: Embedding of the training data in low-dimensional space.
        data2d = self.__tsne.fit_transform(X=self.__matrix)

        print("TSNE embedding shape", data2d.shape, self.__tsne.embedding_.shape)
        #plot
        #plt.scatter(tsne_data2d[:, 0],
        #            tsne_data2d [:, 1],
        #            c = self.__clusters.kmeans_clusters_pred,
        #            cmap=plt.cm.Spectral)
        #plt.show()

        #array-like, shape (n_samples, n_components)
        n_components = self.__tsne.embedding_.shape[1]

        print("Number of TSNE components", n_components)

        y = self.__clusters.kmeans_clusters_pred
        pairs = list(range(0, n_components))
        print(pairs)
        print(pairs[1:] + pairs[:1])
        for i,j in zip(pairs, pairs[1:] + pairs[:1]):
            print("TSNE components", i,j)

            #plot
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
            ax.set_title('TSNE')

            #plt.scatter(data2d[:, i],
            #            data2d[:, j],
            #            c = self.__kmeans_clusters_pred) 

            for icluster in range(self.__clusters.n_actual_clusters):
                print(f"Cluster {icluster}/{self.__clusters.n_actual_clusters}")
                ax.scatter(np.array(data2d[y==icluster, i]),
                           np.array(data2d[y==icluster, j]),
                           s = 100,
                           #c = i,
                           alpha=0.5,
                           label = f"Cluster {icluster}")


            #plt.scatter(pca_cluster_centers2d[:, i],
            #            pca_cluster_centers2d[:, j], 
            #            marker='x', s=200, linewidths=3, c='r')

            ax.set_xlabel(f"component {i}")
            ax.set_ylabel(f"component {j}")
            ax.legend(loc="best")
            plt.show()

    def __fit_mds(self):
        #Fit X data
        mds_data2d = self.__mds.fit_transform(X=self.__matrix)

        #plot
        plt.scatter(mds_data2d[:, 0],
                    mds_data2d [:, 1],
                    c = self.__kmeans_clusters_pred,
                    cmap=plt.cm.Spectral)
        plt.show()

    def __fit_isomap(self):
        #Compute the embedding vectors for data X
        embed = self.__isomap.fit_transform(X=self.__matrix)
        
        #Semantic labeling of cluster.
        #Apply a label if the clusters max TF-IDF is in the 90% quantile
        #of the whole corpus of TF-IDF scores

        #clusterLabels = []
        #t99 = scipy.stats.mstats.mquantiles(self.__matrix, [ 0.9])[0]

        #for i in range(0, vectorized.shape[0]):
        #    row = vectorized.getrow(i)

        #    if row.max() >= t99:
        #        arrayIndex = numpy.where(row.data == row.max())[0][0]
        #        clusterLabels.append(labels[row.indices[arrayIndex]])
        #    else:
        #        clusterLabels.append('')

        # Plot the dimension reduced data

        plt.xlabel('reduced dimension-1')
        plt.ylabel('reduced dimension-2')
        for i in range(1, len(embed)):
            plt.scatter(embed[i][0],
                        embed[i][1])
                        #c=self.__kmeans_clusters_pred)
            #plt.annotate(clusterLabels[i],
            #             embed[i],
            #             xytext=None, xycoords='data', textcoords='data', arrowprops=None)

        plt.show()

    def __display_tfidf(self):

        print('TF-IDF matrix')
        print(self.__tfidf_dataframe)

        #aggregations and ascending order
        estimates = {'mean':False, 'sum':False, 'max':False}
        #print(list(estimates.keys()))
        #print(estimates.values())

        #df = self.__tfidf_dataframe.mean(axis=0)
        df = self.__tfidf_dataframe.agg(list(estimates.keys())).T

        df = df.sort_values(by = list(estimates.keys()),
                            ascending = list(estimates.values()))
        df.index.name = 'term'

        title = 'tf-idf'
        N=20
        if N > 0:
            df = df.head(N) # same as df[:N]
            df = df.tail(N) # same as df[-N:]
            title += f' ({N} top terms)'
        #plt.figure(figsize=(10, 6))
        ax = df[estimates.keys()].plot(kind='bar',
                                       title = title,
                                       figsize=(15, 10),
                                       legend=True,
                                       fontsize=12)
        ax.set_xlabel("term", fontsize=12)
        ax.set_ylabel("score", fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.show()

        #store in txt file
        #print(df);

    def __display_angle_similarity(self):
        df=self.__tfidf_angle_similarity_dataframe
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        vmin = df.where(df>0).min().min()
        vmax=df.max().max() #df.values.max()
        midpoint = (vmax - vmin) / 2
        #smoother washed-out contours
        alpha = 0.10
        vmin_new = vmin*(1-alpha)
        vmax_new = vmax*(1+alpha)
        vmin = vmin_new
        vmax = vmax_new if vmax_new < 90. else 90
        print(f"Angle similarity min {vmin}, max {vmax}")
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111)
        ax.set_title(f'TF-IDF document angle similarity)')
        sns.heatmap(df,
                    mask=mask,
                    annot=True,
                    square=True, #forces the aspect ratio of the blocks to be equal
                    fmt=".1f",
                    vmin=vmin,
                    vmax=vmax,
                    #center=midpoint,
                    cmap="coolwarm",
                    annot_kws={'size':10},
                    ax=ax)
        
        plt.show()

    def fit(self):
        """
        user callable method to invoke fits
        """
        self.__display_tfidf()
        self.__display_angle_similarity()

        print(type(self.__feature_names))
        self.__clusters.fit(matrix = self.__matrix,
                            dataframe = self.__tfidf_dataframe,
                            corpora = self.__corpora_names,
                            features = self.__feature_names)

        self.__fit_pca()
        #self.__fit_lsa()
        self.__fit_tsne()
        #self.__fit_mds()
        #self.__fit_isomap()
        

        
        

        



        


        
