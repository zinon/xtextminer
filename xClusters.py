from sklearn.cluster import KMeans
from scipy.sparse.csr import csr_matrix
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class xClusters:
    """
    Term clusterization
    """
    def __init__(self, **kwargs):
        #input parameters
        self.__n_clusters = kwargs.get('n_clusters', 2)
        self.__kmeans_max_iter = kwargs.get('kmeans_max_iter', 300)

        #tf-idf matrix
        self.__matrix = None

        #tf-idf dataframe
        self.__tfidf_dataframe = None
        
        #feature names
        self.__feature_names = None

        self.__corpora_names = None
        
        #clustered terms
        self.__clustered_terms_dataframe = pd.DataFrame()

        #actual number of clusters after optimization
        #TBD
        self.__n_actual_clusters = self.__n_clusters

        #kmeans class
        self.__kmeans = KMeans(n_clusters = self.__n_clusters,
                               init='k-means++',
                               n_init=1,
                               max_iter = self.__kmeans_max_iter,
                               precompute_distances = "auto",
                               n_jobs= -1,
                               random_state = 33)

        self.__kmeans_clusters = None

        self.__kmeans_cluster_centers = None
        
        self.__kmeans_clusters_pred = None

        self.__kmeans_labels = None

    def __set_properties(self,
                         matrix=None,
                         dataframe:pd.DataFrame()=None,
                         corpora:List[str]=None,
                         features:List[str]=None):
        
        if matrix.size:
            self.__matrix = matrix.todense() \
                if type(matrix) == csr_matrix \
                else matrix

        if len(features):
            self.__feature_names = features

        self.__tfidf_dataframe = dataframe

        if len(corpora):
            self.__corpora_names = corpora

    @property            
    def kmeans_clusters(self):
        return self.__kmeans_clusters

    @property            
    def kmeans_cluster_centers(self):
        return self.__kmeans_cluster_centers
            
    @property
    def kmeans_clusters_pred(self):
        return self.__kmeans_clusters_pred

    @property
    def n_actual_clusters(self):
        return self.__n_actual_clusters

    @property
    def clustered_terms_dataframe(self):
        """
        on-demand creation of dataframe with clustered terms 
        """
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

    def __top_keywords(self, clusters, n_terms:int=10):
        """
        top keywords per cluster
        assumes that matrix is already dense
        """
        df = pd.DataFrame(self.__matrix).groupby(clusters).mean()
    
        for i, r in df.iterrows():
            print(f'Cluster {i}: ' +
                  ', '.join( [self.__feature_names[t] for t in np.argsort(r)[-n_terms:]] ) )

    def __top_terms(self, clusters, centers, n_terms:int=10):
        """
        top terms per cluster
        """
        order_centroids = centers.argsort()[:, ::-1]
        for i in range(clusters):
            top_terms = [self.__feature_names[ind] for ind in order_centroids[i, :n_terms]]
            print("Cluster {}: {}".format(i, ', '.join(top_terms)))

    def __set_clustered_terms_dataframe(self):
        """
        creates a panda DataFrame of corpora associated to clusters
        """
        self.__clustered_terms_dataframe = pd.DataFrame( {'corpus':self.__corpora_names,
                                                          'cluster':self.__kmeans_labels} )

        
    def __optimize_kmeans(self):
        """
        minimize within cluster sum of square (WCSS)
        """
        wcss = []
        for i in range(1,11):
            km=KMeans(n_clusters=i,
                      init='k-means++',
                      max_iter=300,
                      n_init=10, random_state=0)
            km.fit(X=self.__matrix)
            wcss.append(km.inertia_)

        plt.plot(range(1,11), wcss, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        
    def __fit_kmeans(self):
        """
        tf-idf Vectorizer results are normalized, which makes KMeans behave as
        spherical k-means for better results. 
        """
        #Compute k-means clustering
        self.__kmeans_clusters = self.__kmeans.fit(X=self.__matrix)

        #cluster centers
        self.__kmeans_cluster_centers = self.__kmeans_clusters.cluster_centers_
        
        #Compute cluster centers and predict cluster index for each sample.
        self.__kmeans_clusters_pred = self.__kmeans.fit_predict(X=self.__matrix)

        self.__kmeans_labels = self.__kmeans_clusters.labels_.tolist()

        self.__kmeans_cluster_centers = self.__kmeans.cluster_centers_

    def __cluster_top_terms(self):
        #top keywords
        self.__top_keywords(self.__kmeans_clusters_pred, n_terms = 5)

        #Top terms per cluster
        self.__top_terms(self.__n_clusters, self.__kmeans_cluster_centers)
        
    
    def __plot_kmeans(self):
        #CONSIDER TOP N TERMS ONLY: N! / (2! (N-2)!)
        n_selected_columns = 4

        print("TF-IDF DF", self.__tfidf_dataframe.shape)
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

            for icluster in range(self.__n_clusters):
                print("CLUS", icluster,self.__n_clusters)
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
        column_names = self.__tfidf_dataframe.columns.tolist()
        #print(self.__tfidf_dataframe);exit(1)
        
        n_column_names = len(column_names)
        #clusters
        self.__tfidf_dataframe['cluster'] = self.__kmeans_labels

        selected_column_names = column_names[:n_selected_columns]

        #sns.set()
        cmap = sns.cubehelix_palette(dark=.8, light=.3, as_cmap=True)
        for iname, jname in zip(selected_column_names, selected_column_names[1:]):
            print(f"kmeans cluster {iname} {jname}")
            
            fig = plt.figure(figsize=(10,7))
            ax = fig.add_subplot(111)
            ax.set_title(f'TF-IDF ({n_selected_columns} selected documents)')

            #lm = sns.lmplot(data=self.__tfidf_dataframe,
            #            x=iname,
            #            y=jname,
            #            hue='cluster',
            #            fit_reg=False, legend=True, legend_out=True)

            ax = sns.scatterplot(x=iname,
                                 y=jname,
                                 hue='cluster',
                                 size='cluster',
                                 #palette='Set2',
                                 palette=cmap,
                                 data=self.__tfidf_dataframe,
                                 ax=ax)

            #self.__tfidf_dataframe.groupby("cluster").scatter(x=iname,
            #                                                   y=jname,
            #                                                   marker="o",
            #                                                   ax=ax)
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

        xscatter = self.__tfidf_dataframe.plot(kind='scatter',
                                          x=column_names[0],
                                          y=column_names[1],
                                          alpha=0.1,
                                          s=300,
                                          cmap='viridis',
                                          c=y)
        '''

    def fit(self,
            matrix = None,
            dataframe:pd.DataFrame() = None,
            corpora:List[str] = None,
            features:List[str] = None):
        """
        user callable method
        """
        self.__set_properties(matrix, dataframe, corpora, features)
        self.__optimize_kmeans()
        self.__fit_kmeans()
        self.__cluster_top_terms()
        self.__plot_kmeans()

