NB. jlinreg.ijs — Linear Regression coclass
NB. Usage:
NB.   m =. conew 'jlinreg'
NB.   fit__m data ; responses           NB. or lambda fit__m data ; responses
NB.   betas__m ''                       NB. returns intercept , coef1 , coef2 , ...
NB.   preds =. predict__m testdata
NB.   destroy__m ''

NB. Load jmlpack FFI layer relative to this file
JLINREG_DIR =: ({.~ i:&'/') > (4!:4 <'JLINREG_DIR') { 4!:3 ''
require JLINREG_DIR , '/jmlpack.ijs'

coclass 'jlinreg'

create =: 3 : 0
  handle =: > 0 { linreg_create_jmlpack_ ''
)

NB. fit: y is data;responses. x (optional) is lambda.
NB. data is (n_obs, n_feat) matrix, responses is n_obs vector.
fit =: 3 : 0
  0 fit y
:
  'data responses' =. y
  lambda =. x
  n_obs =. 0 { $ data
  nfeat =: 1 { $ data
  rc =. > 0 { linreg_train_jmlpack_ handle ; (, data) ; n_obs ; nfeat ; responses ; lambda
  assert rc = 0
  i. 0 0
)

NB. betas: returns intercept , coef1 , coef2 , ...
betas =: 3 : 0
  buf =. (nfeat + 1) $ 0.0
  r =. linreg_betas_jmlpack_ handle ; nfeat ; buf
  > 3 { r
)

predict =: 3 : 0
  data =. y
  n_test =. 0 { $ data
  n_feat =. 1 { $ data
  buf =. n_test $ 0.0
  r =. linreg_predict_jmlpack_ handle ; (, data) ; n_test ; n_feat ; buf
  > 5 { r
)

destroy =: 3 : 0
  linreg_destroy_jmlpack_ < handle
  codestroy ''
)

cocurrent 'base'
