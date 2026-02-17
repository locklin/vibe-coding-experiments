NB. jlogreg.ijs — Logistic Regression coclass
NB. Usage:
NB.   m =. conew 'jlogreg'
NB.   fit__m data ; labels              NB. or lambda fit__m data ; labels
NB.   betas__m ''                       NB. returns intercept , coef1 , coef2 , ...
NB.   preds =. classify__m testdata
NB.   destroy__m ''

NB. Load jmlpack FFI layer relative to this file
JLOGREG_DIR =: ({.~ i:&'/') > (4!:4 <'JLOGREG_DIR') { 4!:3 ''
require JLOGREG_DIR , '/jmlpack.ijs'

coclass 'jlogreg'

create =: 3 : 0
  handle =: > 0 { logreg_create_jmlpack_ ''
)

NB. fit: y is data;labels. x (optional) is lambda.
NB. data is (n_obs, n_feat) matrix, labels is n_obs integer vector.
fit =: 3 : 0
  0 fit y
:
  'data labels' =. y
  lambda =. x
  n_obs =. 0 { $ data
  nfeat =: 1 { $ data
  rc =. > 0 { logreg_train_jmlpack_ handle ; (, data) ; n_obs ; nfeat ; labels ; lambda
  assert rc = 0
  i. 0 0
)

NB. betas: returns intercept , coef1 , coef2 , ...
betas =: 3 : 0
  buf =. (nfeat + 1) $ 0.0
  r =. logreg_betas_jmlpack_ handle ; nfeat ; buf
  > 3 { r
)

classify =: 3 : 0
  data =. y
  n_test =. 0 { $ data
  n_feat =. 1 { $ data
  buf =. n_test $ 00
  r =. logreg_classify_jmlpack_ handle ; (, data) ; n_test ; n_feat ; buf
  > 5 { r
)

destroy =: 3 : 0
  logreg_destroy_jmlpack_ < handle
  codestroy ''
)

cocurrent 'base'
