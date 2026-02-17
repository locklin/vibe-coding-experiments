NB. jknn.ijs — K-Nearest Neighbors coclass
NB. Usage:
NB.   m =. conew 'jknn'
NB.   train__m refdata
NB.   'neighbors distances' =. k search__m querydata
NB.   destroy__m ''

NB. Load jmlpack FFI layer relative to this file
JKNN_DIR =: ({.~ i:&'/') > (4!:4 <'JKNN_DIR') { 4!:3 ''
require JKNN_DIR , '/jmlpack.ijs'

coclass 'jknn'

create =: 3 : 0
  handle =: > 0 { knn_create_jmlpack_ ''
)

NB. train: y is reference data (n_obs, n_feat)
train =: 3 : 0
  data =. y
  n_obs =. 0 { $ data
  n_feat =. 1 { $ data
  rc =. > 0 { knn_train_jmlpack_ handle ; (, data) ; n_obs ; n_feat
  assert rc = 0
  i. 0 0
)

NB. search: y is query data (n_query, n_feat). x is k.
NB. Returns neighbors;distances, each (n_query, k).
search =: 4 : 0
  query =. y
  k =. x
  n_query =. 0 { $ query
  n_feat =. 1 { $ query
  nbuf =. (n_query * k) $ 00
  dbuf =. (n_query * k) $ 0.0
  NB. knn_search: i x *d i i i *i *d
  NB. Result: 0=status 1=handle 2=query 3=n_query 4=n_feat 5=k 6=neighbors 7=distances
  r =. knn_search_jmlpack_ handle ; (, query) ; n_query ; n_feat ; k ; nbuf ; dbuf
  ((n_query , k) $ > 6 { r) ; (n_query , k) $ > 7 { r
)

destroy =: 3 : 0
  knn_destroy_jmlpack_ < handle
  codestroy ''
)

cocurrent 'base'
