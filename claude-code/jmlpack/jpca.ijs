NB. jpca.ijs — PCA coclass
NB. Usage:
NB.   m =. conew 'jpca'           NB. or 1 conew 'jpca' for scaled
NB.   'transformed eigvals varret' =. new_dim apply__m data
NB.   destroy__m ''

NB. Load jmlpack FFI layer relative to this file
JPCA_DIR =: ({.~ i:&'/') > (4!:4 <'JPCA_DIR') { 4!:3 ''
require JPCA_DIR , '/jmlpack.ijs'

coclass 'jpca'

NB. create: y is scale_data flag (0 or 1, default 0)
create =: 3 : 0
  handle =: > 0 { pca_create_jmlpack_ < y
)

NB. apply: y is data (n_obs, n_feat). x is new_dim.
NB. Returns transformed;eigenvalues;variance_retained.
apply =: 4 : 0
  data =. y
  new_dim =. x
  n_obs =. 0 { $ data
  n_feat =. 1 { $ data
  obuf =. (n_obs * new_dim) $ 0.0
  ebuf =. new_dim $ 0.0
  vbuf =. 1 $ 0.0
  NB. pca_apply: i x *d i i *d i *d *d
  NB. Result: 0=status 1=handle 2=data 3=n_obs 4=n_feat 5=output 6=new_dim 7=eigvals 8=varret
  r =. pca_apply_jmlpack_ handle ; (, data) ; n_obs ; n_feat ; obuf ; new_dim ; ebuf ; vbuf
  ((n_obs , new_dim) $ > 5 { r) ; (> 7 { r) ; (> 8 { r)
)

destroy =: 3 : 0
  pca_destroy_jmlpack_ < handle
  codestroy ''
)

cocurrent 'base'
