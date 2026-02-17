NB. jmlpack.ijs — J FFI layer for libjmlpack.so
NB. Provides 15!:0 bonded wrappers for all C functions.
NB.
NB. Usage pattern: each function is 'declaration'&(15!:0)
NB. Called monadically: func arg1 ; arg2 ; ...
NB. For no-arg functions: func ''
NB. Returns boxed list: item 0 is return value, then each arg.

cocurrent 'jmlpack'

NB. Resolve library path: same directory as this script
JMLPACK_DIR =: ({.~ i:&'/') > (4!:4 <'JMLPACK_DIR') { 4!:3 ''
JMLPACK_LIB =: JMLPACK_DIR , '/libjmlpack.so'

NB. ============================================================
NB. Linear Regression
NB. ============================================================

linreg_create =: (JMLPACK_LIB , ' jmlpack_linreg_create x')&(15!:0)

linreg_train =: (JMLPACK_LIB , ' jmlpack_linreg_train i x *d i i *d d')&(15!:0)

linreg_predict =: (JMLPACK_LIB , ' jmlpack_linreg_predict i x *d i i *d')&(15!:0)

linreg_betas =: (JMLPACK_LIB , ' jmlpack_linreg_betas i x i *d')&(15!:0)

linreg_destroy =: (JMLPACK_LIB , ' jmlpack_linreg_destroy n x')&(15!:0)

NB. ============================================================
NB. Logistic Regression
NB. ============================================================

logreg_create =: (JMLPACK_LIB , ' jmlpack_logreg_create x')&(15!:0)

logreg_train =: (JMLPACK_LIB , ' jmlpack_logreg_train i x *d i i *i d')&(15!:0)

logreg_classify =: (JMLPACK_LIB , ' jmlpack_logreg_classify i x *d i i *i')&(15!:0)

logreg_betas =: (JMLPACK_LIB , ' jmlpack_logreg_betas i x i *d')&(15!:0)

logreg_destroy =: (JMLPACK_LIB , ' jmlpack_logreg_destroy n x')&(15!:0)

NB. ============================================================
NB. Decision Tree
NB. ============================================================

dtree_create =: (JMLPACK_LIB , ' jmlpack_dtree_create x')&(15!:0)

dtree_train =: (JMLPACK_LIB , ' jmlpack_dtree_train i x *d i i *i i i i')&(15!:0)

dtree_classify =: (JMLPACK_LIB , ' jmlpack_dtree_classify i x *d i i *i')&(15!:0)

dtree_destroy =: (JMLPACK_LIB , ' jmlpack_dtree_destroy n x')&(15!:0)

NB. ============================================================
NB. Random Forest
NB. ============================================================

rforest_create =: (JMLPACK_LIB , ' jmlpack_rforest_create x')&(15!:0)

rforest_train =: (JMLPACK_LIB , ' jmlpack_rforest_train i x *d i i *i i i i i')&(15!:0)

rforest_classify =: (JMLPACK_LIB , ' jmlpack_rforest_classify i x *d i i *i')&(15!:0)

rforest_destroy =: (JMLPACK_LIB , ' jmlpack_rforest_destroy n x')&(15!:0)

NB. ============================================================
NB. K-Means
NB. ============================================================

kmeans_create =: (JMLPACK_LIB , ' jmlpack_kmeans_create x')&(15!:0)

kmeans_cluster =: (JMLPACK_LIB , ' jmlpack_kmeans_cluster i x *d i i i *i *d')&(15!:0)

kmeans_destroy =: (JMLPACK_LIB , ' jmlpack_kmeans_destroy n x')&(15!:0)

NB. ============================================================
NB. KNN
NB. ============================================================

knn_create =: (JMLPACK_LIB , ' jmlpack_knn_create x')&(15!:0)

knn_train =: (JMLPACK_LIB , ' jmlpack_knn_train i x *d i i')&(15!:0)

knn_search =: (JMLPACK_LIB , ' jmlpack_knn_search i x *d i i i *i *d')&(15!:0)

knn_destroy =: (JMLPACK_LIB , ' jmlpack_knn_destroy n x')&(15!:0)

NB. ============================================================
NB. PCA
NB. ============================================================

pca_create =: (JMLPACK_LIB , ' jmlpack_pca_create x i')&(15!:0)

pca_apply =: (JMLPACK_LIB , ' jmlpack_pca_apply i x *d i i *d i *d *d')&(15!:0)

pca_destroy =: (JMLPACK_LIB , ' jmlpack_pca_destroy n x')&(15!:0)

cocurrent 'base'
