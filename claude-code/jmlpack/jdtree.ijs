NB. jdtree.ijs — Decision Tree coclass
NB. Usage:
NB.   m =. conew 'jdtree'
NB.   (n_classes;min_leaf;max_depth) fit__m data ; labels
NB.   preds =. classify__m testdata
NB.   destroy__m ''

NB. Load jmlpack FFI layer relative to this file
JDTREE_DIR =: ({.~ i:&'/') > (4!:4 <'JDTREE_DIR') { 4!:3 ''
require JDTREE_DIR , '/jmlpack.ijs'

coclass 'jdtree'

create =: 3 : 0
  handle =: > 0 { dtree_create_jmlpack_ ''
)

NB. fit: y is data;labels. x is n_classes;min_leaf;max_depth (or defaults).
NB. data is (n_obs, n_feat) matrix, labels is n_obs integer vector.
fit =: 3 : 0
  (2;1;10) fit y
:
  'data labels' =. y
  'n_classes min_leaf max_depth' =. x
  n_obs =. 0 { $ data
  n_feat =. 1 { $ data
  rc =. > 0 { dtree_train_jmlpack_ handle ; (, data) ; n_obs ; n_feat ; labels ; n_classes ; min_leaf ; max_depth
  assert rc = 0
  i. 0 0
)

classify =: 3 : 0
  data =. y
  n_test =. 0 { $ data
  n_feat =. 1 { $ data
  buf =. n_test $ 00
  r =. dtree_classify_jmlpack_ handle ; (, data) ; n_test ; n_feat ; buf
  > 5 { r
)

destroy =: 3 : 0
  dtree_destroy_jmlpack_ < handle
  codestroy ''
)

cocurrent 'base'
