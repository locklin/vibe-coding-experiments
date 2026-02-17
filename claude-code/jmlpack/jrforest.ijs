NB. jrforest.ijs — Random Forest coclass
NB. Usage:
NB.   m =. conew 'jrforest'
NB.   (n_classes;n_trees;min_leaf;max_depth) fit__m data ; labels
NB.   preds =. classify__m testdata
NB.   destroy__m ''

NB. Load jmlpack FFI layer relative to this file
JRFOREST_DIR =: ({.~ i:&'/') > (4!:4 <'JRFOREST_DIR') { 4!:3 ''
require JRFOREST_DIR , '/jmlpack.ijs'

coclass 'jrforest'

create =: 3 : 0
  handle =: > 0 { rforest_create_jmlpack_ ''
)

NB. fit: y is data;labels. x is n_classes;n_trees;min_leaf;max_depth (or defaults).
NB. data is (n_obs, n_feat) matrix, labels is n_obs integer vector.
fit =: 3 : 0
  (2;10;1;10) fit y
:
  'data labels' =. y
  'n_classes n_trees min_leaf max_depth' =. x
  n_obs =. 0 { $ data
  n_feat =. 1 { $ data
  rc =. > 0 { rforest_train_jmlpack_ handle ; (, data) ; n_obs ; n_feat ; labels ; n_classes ; n_trees ; min_leaf ; max_depth
  assert rc = 0
  i. 0 0
)

classify =: 3 : 0
  data =. y
  n_test =. 0 { $ data
  n_feat =. 1 { $ data
  buf =. n_test $ 00
  r =. rforest_classify_jmlpack_ handle ; (, data) ; n_test ; n_feat ; buf
  > 5 { r
)

destroy =: 3 : 0
  rforest_destroy_jmlpack_ < handle
  codestroy ''
)

cocurrent 'base'
