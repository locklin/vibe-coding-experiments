NB. jkmeans.ijs — K-Means Clustering coclass
NB. Usage:
NB.   m =. conew 'jkmeans'
NB.   'assignments centroids' =. k cluster__m data
NB.   destroy__m ''

NB. Load jmlpack FFI layer relative to this file
JKMEANS_DIR =: ({.~ i:&'/') > (4!:4 <'JKMEANS_DIR') { 4!:3 ''
require JKMEANS_DIR , '/jmlpack.ijs'

coclass 'jkmeans'

create =: 3 : 0
  handle =: > 0 { kmeans_create_jmlpack_ ''
)

NB. cluster: y is data (n_obs, n_feat). x is k.
NB. Returns assignments;centroids where centroids is (k, n_feat).
cluster =: 4 : 0
  data =. y
  k =. x
  n_obs =. 0 { $ data
  n_feat =. 1 { $ data
  abuf =. n_obs $ 00
  cbuf =. (k * n_feat) $ 0.0
  NB. kmeans_cluster: i x *d i i i *i *d
  NB. Result indices: 0=status 1=handle 2=data 3=n_obs 4=n_feat 5=k 6=assignments 7=centroids
  r =. kmeans_cluster_jmlpack_ handle ; (, data) ; n_obs ; n_feat ; k ; abuf ; cbuf
  (> 6 { r) ; (k , n_feat) $ > 7 { r
)

destroy =: 3 : 0
  kmeans_destroy_jmlpack_ < handle
  codestroy ''
)

cocurrent 'base'
