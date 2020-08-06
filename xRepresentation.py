from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.sparse.csr import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from typing import List
import seaborn as sns

class xRepresentation:
    """
    Perform dimensionality reduction and represent 2d matrices 
    """
    def __init__(self, **kwargs):
        self.__clusters = kwargs.get('clusters', 2)
        print("hELLO", self.__clusters)
        self.__pca_components = kwargs.get('pca_components', 3)
        self.__components = kwargs.get('components', 2)
        self.__kmeans_max_iter = kwargs.get('kmeans_max_iter', 300)
        self.__neighbors = kwargs.get('neighbors', 2) #10
        self.__lsa_normalization = False

        #sparse matrix
        self.__matrix = None
        #tf-idf dataframe
        self.__data_frame_tfidf = pd.DataFrame()
        self.__corpora_names = None
        self.__feature_names = None
        #clustered terms
        self.__clustered_terms_dataframe = pd.DataFrame()

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
        
        #kmeans
        self.__kmeans = KMeans(n_clusters = self.__clusters,
                               init='k-means++',
                               n_init=1,
                               max_iter = self.__kmeans_max_iter,
                               precompute_distances = "auto",
                               n_jobs= -1,
                               random_state = 33)

        self.__kmeans_clusters = None
        self.__kmeans_clusters_pred = None
        self.__kmeans_labels = None

        #cluster centers
        self.__kmeans_cluster_centers = None
        
        #normalizer
        self.__normalizer = Normalizer(copy=False)


    @property
    def matrix(self) -> np.matrix:
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix:csr_matrix):
        if matrix.size:
            self.__matrix = matrix.todense()

    #@property
    #def data_frame_tfidf(self) -> pd.DataFrame():
    #    return self.__data_frame_tfidf

    #@data_frame_tfidf.setter
    #def data_frame_tfidf(self, df:pd.DataFrame() = None):
    #    self.__data_frame_tfidf = df

    def set_data_frame_tfidf(self, df:pd.DataFrame()):
        self.__data_frame_tfidf = df
        
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
    def clustered_terms_dataframe(self):
        if self.__none_clustered_terms_dataframe():
            self.__set_clustered_terms_dataframe()
        return self.__clustered_terms_dataframe

    def corpus_cluster(self, corpus_name) -> str:
        """
        user callable method
        returns cluster associated to a given corpus name
        # .values turns the series into a list and [0] selects the single element
        """
        if self.__none_clustered_terms_dataframe():
            self.__set_clustered_terms_dataframe()

        return self.__clustered_terms_dataframe[
            self.__clustered_terms_dataframe['corpus']==corpus_name]['cluster'].values[0]

    def cluster_corpora(self, cluster:int) -> List[str]:
        """
        user callable method
        returns list of corpora names associated to a given cluster
        """
        if self.__none_clustered_terms_dataframe():
            self.__set_clustered_terms_dataframe()

        return self.__clustered_terms_dataframe[
            self.__clustered_terms_dataframe['cluster']==cluster]['corpus'].values.tolist()

    def __none_clustered_terms_dataframe(self):
        return self.__clustered_terms_dataframe.empty
    
    def __normalize_matrix(self):
         transformer = self.__normalizer.fit(self.__matrix) # fit does nothing
         self.__matrix = transformer.transform(self.__matrix)

    def __top_keywords(self, clusters, n_terms):
        #assuming matrix is already dense
        df = pd.DataFrame(self.__matrix).groupby(clusters).mean()
    
        for i, r in df.iterrows():
            print(f'Cluster {i}: ' +
                  ', '.join( [self.__feature_names[t] for t in np.argsort(r)[-n_terms:]] ) )

    def __top_terms(self, clusters, centers):
        order_centroids = centers.argsort()[:, ::-1]
        for i in range(clusters):
            top_terms = [self.__feature_names[ind] for ind in order_centroids[i, :10]]
            print("Cluster {}: {}".format(i, ' '.join(top_terms)))

    def __set_clustered_terms_dataframe(self):
        """
        creates a panda DataFrame of corpora associated to clusters
        """
        self.__clustered_terms_dataframe = pd.DataFrame( {'corpus':self.__corpora_names,
                                                          'cluster':self.__kmeans_labels} )
        
    def __fit_kmeans(self):
        """
        tf-idf Vectorizer results are normalized, which makes KMeans behave as
        spherical k-means for better results. 
        """
        #Compute k-means clustering
        self.__kmeans_clusters = self.__kmeans.fit(X=self.__matrix)

        #Compute cluster centers and predict cluster index for each sample.
        self.__kmeans_clusters_pred = self.__kmeans.fit_predict(X=self.__matrix)

        self.__kmeans_labels = self.__kmeans_clusters.labels_.tolist()

        self.__kmeans_cluster_centers = self.__kmeans.cluster_centers_

        #top keywords
        self.__top_keywords(self.__kmeans_clusters_pred, n_terms = 5)

        #Top terms per cluster
        self.__top_terms(self.__clusters, self.__kmeans_cluster_centers)
        

        self.__plot_kmeans()
        
    def __plot_kmeans(self):
        #CONSIDER TOP N TERMS ONLY: N! / (2! (N-2)!)
        n_selected_columns = 4

        print("TF-IDF DF", self.__data_frame_tfidf.shape)
        print("Sparse matrix", self.__matrix.shape)

        ##Note: to avoid "ValueError: Masked arrays must be 1-D"
        ##      cast the data into an numpy array.
        y = self.__kmeans_clusters_pred
        L = list(range(0, n_selected_columns))
        for i,j in zip(L, L[1:] + L[:1]):        
            print(f"TF-IDF columns {i}:{j}")

            #plot matrix clusters pairwise
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
            ax.set_title(f'Clusters ({n_selected_columns} selected documents)')

            for icluster in range(self.__clusters):
                print("CLUS", icluster,self.__clusters)
                ax.scatter(np.array(self.__matrix[y==icluster, i]),
                           np.array(self.__matrix[y==icluster, j]),
                           s = 100,
                           #c = i,
                           alpha=0.1,
                           label = f"Cluster {icluster}")

            ax.scatter(self.__kmeans_cluster_centers[:, i],
                       self.__kmeans_cluster_centers[:, j], 
                       marker='x',
                       s=200,
                       linewidths=3,
                       c='r')

            ax.set_xlabel(f"doc {i}")
            ax.set_ylabel(f"doc {j}")
            ax.legend(loc="best")
            plt.show()

        """
        ax.scatter(np.array(self.__matrix[:, i]),
        np.array(self.__matrix[:, j]),
        c=y,
        s=50,
        cmap='viridis')
        """        



        #
        #feature names; before adding cluster information to the df
        column_names = self.__data_frame_tfidf.columns.tolist()
        #print(self.__data_frame_tfidf);exit(1)
        
        n_column_names = len(column_names)
        #clusters
        self.__data_frame_tfidf['cluster'] = self.__kmeans_labels

        selected_column_names = column_names[:n_selected_columns]
        for iname, jname in zip(selected_column_names, selected_column_names[1:]):

            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
            ax.set_title(f'TF-IDF ({n_selected_columns} selected documents)')

            sns.lmplot(data=self.__data_frame_tfidf,
                       x=iname,
                       y=jname,
                       hue='cluster',
                       fit_reg=False, legend=True, legend_out=True)
                       

            ax.set_xlabel(iname)
            ax.set_ylabel(jname)
            ax.legend(loc="best")
            plt.show()

        '''        
        cluster_colors = set(y)
        xscatter= ax.scatter(x=column_names[0],
                             y=column_names[1],
                             c=1,
                             s=300,
                             cmap='viridis')

        xscatter = self.__data_frame_tfidf.plot(kind='scatter',
                                          x=column_names[0],
                                          y=column_names[1],
                                          alpha=0.1,
                                          s=300,
                                          cmap='viridis',
                                          c=y)
        '''


    def __fit_pca(self):
        
        #Fit the model with X.
        pca = self.__pca.fit(X=self.__matrix)

        #Apply dimensionality reduction to X.
        pca_data2d = pca.transform(X=self.__matrix)

        #calculate the cluster enters on the reduced data
        pca_cluster_centers2d = pca.transform(self.__kmeans_clusters.cluster_centers_)

        n_pca_components = self.__pca.components_.shape[0]

        print("Number of PCA components", n_pca_components)


        y = self.__kmeans_clusters_pred
        pairs = list(range(0, n_pca_components))
        for i,j in zip(pairs, pairs[1:] + pairs[:1]):
            print("PCA components", i,j)

            #plot
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
            ax.set_title('PCA')

            #plt.scatter(pca_data2d[:, i],
            #            pca_data2d[:, j],
            #            c = self.__kmeans_clusters_pred) 

            for icluster in range(self.__clusters):
                print("CLUS", icluster,self.__clusters)
                ax.scatter(np.array(pca_data2d[y==icluster, i]),
                           np.array(pca_data2d[y==icluster, j]),
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
        tsne_data2d = self.__tsne.fit_transform(X=self.__matrix)

        #plot
        plt.scatter(tsne_data2d[:, 0],
                    tsne_data2d [:, 1],
                    c = self.__kmeans_clusters_pred,
                    cmap=plt.cm.Spectral)
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

        print('Total')
        print(self.__data_frame_tfidf)

        #aggregations and ascending order
        estimates = {'mean':False, 'sum':False, 'max':False}
        #print(list(estimates.keys()))
        #print(estimates.values())

        #df = self.__data_frame_tfidf.mean(axis=0)
        df = self.__data_frame_tfidf.agg(list(estimates.keys())).T

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
        plt.show()

        #store in txt file
        #print(df);

    
    def fit(self):
        self.__display_tfidf()
        
        self.__fit_kmeans()
        self.__fit_pca()
        #self.__fit_lsa()
        #self.__fit_tsne()
        #self.__fit_mds()
        #self.__fit_isomap()
        

        
        

        



        


        
