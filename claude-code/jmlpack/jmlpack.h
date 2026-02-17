#ifndef JMLPACK_H
#define JMLPACK_H

#ifdef __cplusplus
extern "C" {
#endif

/* Linear Regression */
void* jmlpack_linreg_create(void);
int   jmlpack_linreg_train(void* handle, const double* data, int n_obs, int n_feat,
                           const double* responses, double lambda);
int   jmlpack_linreg_predict(void* handle, const double* data, int n_test, int n_feat,
                             double* predictions_out);
int   jmlpack_linreg_betas(void* handle, int n_feat, double* betas_out);
void  jmlpack_linreg_destroy(void* handle);

/* Logistic Regression */
void* jmlpack_logreg_create(void);
int   jmlpack_logreg_train(void* handle, const double* data, int n_obs, int n_feat,
                           const int* labels, double lambda);
int   jmlpack_logreg_classify(void* handle, const double* data, int n_test, int n_feat,
                              int* predictions_out);
int   jmlpack_logreg_betas(void* handle, int n_feat, double* betas_out);
void  jmlpack_logreg_destroy(void* handle);

/* Decision Tree */
void* jmlpack_dtree_create(void);
int   jmlpack_dtree_train(void* handle, const double* data, int n_obs, int n_feat,
                          const int* labels, int n_classes, int min_leaf, int max_depth);
int   jmlpack_dtree_classify(void* handle, const double* data, int n_test, int n_feat,
                             int* predictions_out);
void  jmlpack_dtree_destroy(void* handle);

/* Random Forest */
void* jmlpack_rforest_create(void);
int   jmlpack_rforest_train(void* handle, const double* data, int n_obs, int n_feat,
                            const int* labels, int n_classes, int n_trees,
                            int min_leaf, int max_depth);
int   jmlpack_rforest_classify(void* handle, const double* data, int n_test, int n_feat,
                               int* predictions_out);
void  jmlpack_rforest_destroy(void* handle);

/* K-Means Clustering */
void* jmlpack_kmeans_create(void);
int   jmlpack_kmeans_cluster(void* handle, const double* data, int n_obs, int n_feat,
                             int k, int* assignments_out, double* centroids_out);
void  jmlpack_kmeans_destroy(void* handle);

/* K-Nearest Neighbors */
void* jmlpack_knn_create(void);
int   jmlpack_knn_train(void* handle, const double* data, int n_obs, int n_feat);
int   jmlpack_knn_search(void* handle, const double* query, int n_query, int n_feat,
                         int k, int* neighbors_out, double* distances_out);
void  jmlpack_knn_destroy(void* handle);

/* PCA */
void* jmlpack_pca_create(int scale_data);
int   jmlpack_pca_apply(void* handle, const double* data, int n_obs, int n_feat,
                        double* output, int new_dim, double* eigvals_out,
                        double* varret_out);
void  jmlpack_pca_destroy(void* handle);

#ifdef __cplusplus
}
#endif

#endif /* JMLPACK_H */
