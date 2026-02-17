NB. examples.ijs — jmlpack demo
NB. Demonstrates linear regression, logistic regression, random forest,
NB. k-means, k-nearest neighbors, and PCA on randomly generated data.
NB.
NB. Run:  jconsole examples.ijs

NB. ── Load all modules ──────────────────────────────────────────────
EXAMPLES_DIR =: ({.~ i:&'/') > (4!:4 <'EXAMPLES_DIR') { 4!:3 ''
require EXAMPLES_DIR , '/jlinreg.ijs'
require EXAMPLES_DIR , '/jlogreg.ijs'
require EXAMPLES_DIR , '/jrforest.ijs'
require EXAMPLES_DIR , '/jkmeans.ijs'
require EXAMPLES_DIR , '/jknn.ijs'
require EXAMPLES_DIR , '/jpca.ijs'

smoutput ''
smoutput '=== jmlpack examples ==='
smoutput ''

NB. ── 1. Linear Regression ──────────────────────────────────────────
NB. Generate 200 observations of y = 3*x1 - 2*x2 + 5 with noise
smoutput '--- 1. Linear Regression ---'
smoutput 'Model: y = 3*x1 - 2*x2 + 5  (with small noise)'
smoutput ''
n =: 200
x1 =: n ?@$ 0
x2 =: n ?@$ 0
X =: x1 ,. x2
y =: (3 * x1) + (_2 * x2) + 5 + 0.01 * (n ?@$ 0) - 0.5

m =: '' conew 'jlinreg'
fit__m X ; y

NB. Print learned coefficients
b =: betas__m ''
smoutput 'Learned coefficients (true: intercept=5, x1=3, x2=-2):'
smoutput '  intercept: ' , ": 0 { b
smoutput '  x1:        ' , ": 1 { b
smoutput '  x2:        ' , ": 2 { b
smoutput ''

NB. Predict on 5 test points
Xtest =: 5 2 $ 0 0 1 0 0 1 1 1 0.5 0.5
expected =: (3 * 0 { |: Xtest) + (_2 * 1 { |: Xtest) + 5
preds =: predict__m Xtest
errors =: | preds - expected
destroy__m ''

fmtrow =: 4 : 0
  'e p r' =. y
  '  ' , (8j4 ": 0{x) , '  ' , (8j4 ": 1{x) , '  ' , (8j4 ": e) , '  ' , (8j4 ": p) , '  ' , 8j4 ": r
)
smoutput '  x1        x2        expected  predicted error'
smoutput '  --------  --------  --------  --------- ---------'
smoutput (0{Xtest) fmtrow (0{expected) , (0{preds) , 0{errors
smoutput (1{Xtest) fmtrow (1{expected) , (1{preds) , 1{errors
smoutput (2{Xtest) fmtrow (2{expected) , (2{preds) , 2{errors
smoutput (3{Xtest) fmtrow (3{expected) , (3{preds) , 3{errors
smoutput (4{Xtest) fmtrow (4{expected) , (4{preds) , 4{errors
smoutput ''
smoutput 'Max prediction error: ' , ": >./ errors
smoutput ''

NB. ── 2. Logistic Regression ────────────────────────────────────────
NB. Two clusters: class 0 near (-3,-3), class 1 near (3,3)
smoutput '--- 2. Logistic Regression ---'
smoutput 'Training: 150 class-0 points near (-3,-3), 150 class-1 points near (3,3)'
smoutput ''
n =: 150
c0x =: _3 + n ?@$ 0
c0y =: _3 + n ?@$ 0
c1x =:  3 + n ?@$ 0
c1y =:  3 + n ?@$ 0
Xtrain =: (c0x , c1x) ,. (c0y , c1y)
labels =: (n $ 0) , (n $ 1)

m =: '' conew 'jlogreg'
fit__m Xtrain ; labels

NB. Print learned coefficients
b =: betas__m ''
smoutput 'Learned coefficients:'
smoutput '  intercept (bias): ' , ": 0 { b
smoutput '  x1:               ' , ": 1 { b
smoutput '  x2:               ' , ": 2 { b
smoutput '  (positive coefs mean feature pushes toward class 1)'
smoutput ''

Xtest =: 4 2 $ _5 _5 _2 _2 2 2 5 5
expected =: 0 0 1 1
preds =: classify__m Xtest
destroy__m ''

smoutput '  Test point    Expected  Predicted'
smoutput '  ----------    --------  ---------'
smoutput '  (-5, -5)      0         ' , ": 0 { preds
smoutput '  (-2, -2)      0         ' , ": 1 { preds
smoutput '  ( 2,  2)      1         ' , ": 2 { preds
smoutput '  ( 5,  5)      1         ' , ": 3 { preds
smoutput ''
smoutput 'Accuracy: ' , (": <. 100 * (+/ preds = expected) % # expected) , '%'
smoutput ''

NB. ── 3. Random Forest ──────────────────────────────────────────────
NB. Four-quadrant XOR pattern:
NB.   Q1 (+,+) -> 0   Q2 (-,+) -> 1   Q3 (-,-) -> 0   Q4 (+,-) -> 1
smoutput '--- 3. Random Forest (20 trees) ---'
smoutput 'XOR-quadrant pattern: class = (x>0) XOR (y>0)'
smoutput '  Q1 (+,+) -> 0   Q2 (-,+) -> 1'
smoutput '  Q3 (-,-) -> 0   Q4 (+,-) -> 1'
smoutput ''
n =: 150
q1x =:  2 + n ?@$ 0 [ q1y =:  2 + n ?@$ 0
q2x =: _3 + n ?@$ 0 [ q2y =:  2 + n ?@$ 0
q3x =: _3 + n ?@$ 0 [ q3y =: _3 + n ?@$ 0
q4x =:  2 + n ?@$ 0 [ q4y =: _3 + n ?@$ 0
Xtrain =: (q1x , q2x , q3x , q4x) ,. (q1y , q2y , q3y , q4y)
labels =: (n $ 0) , (n $ 1) , (n $ 0) , (n $ 1)

m =: '' conew 'jrforest'
(2;20;1;20) fit__m Xtrain ; labels
smoutput 'Trained on ' , (": 4 * n) , ' points (', (": n) , ' per quadrant)'

Xtest =: 4 2 $ 2.5 2.5 _2.5 2.5 _2.5 _2.5 2.5 _2.5
expected =: 0 1 0 1
preds =: classify__m Xtest
destroy__m ''

smoutput ''
smoutput '  Quadrant   Point         Expected  Predicted'
smoutput '  --------   -----         --------  ---------'
smoutput '  Q1 (+,+)   ( 2.5,  2.5)  0         ' , ": 0 { preds
smoutput '  Q2 (-,+)   (-2.5,  2.5)  1         ' , ": 1 { preds
smoutput '  Q3 (-,-)   (-2.5, -2.5)  0         ' , ": 2 { preds
smoutput '  Q4 (+,-)   ( 2.5, -2.5)  1         ' , ": 3 { preds
smoutput ''
smoutput 'Accuracy: ' , (": <. 100 * (+/ preds = expected) % # expected) , '%'
smoutput ''

NB. ── 4. K-Means Clustering ─────────────────────────────────────────
NB. Three clusters at (0,0), (50,50), (100,0)
smoutput '--- 4. K-Means Clustering ---'
smoutput 'Three clusters of 80 points each, true centers at:'
smoutput '  A: (  0.5,  0.5)   B: (50.5, 50.5)   C: (100.5, 0.5)'
smoutput ''
n =: 80
ax =:  0 + n ?@$ 0 [ ay =:  0 + n ?@$ 0
bx =: 50 + n ?@$ 0 [ by =: 50 + n ?@$ 0
cx =: 100 + n ?@$ 0 [ cy =: 0 + n ?@$ 0
Xdata =: (ax , bx , cx) ,. (ay , by , cy)

m =: '' conew 'jkmeans'
'assignments centroids' =: 3 cluster__m Xdata
destroy__m ''

NB. Sort centroids by x-coordinate for consistent display
ord =: /: 0 { |: centroids
sorted =: ord { centroids
sizes =: +/ assignments =/ ord

smoutput 'Found centroids (sorted by x):'
smoutput '  Cluster 0: (' , (6j2 ": (<0;0){sorted) , ', ' , (6j2 ": (<0;1){sorted) , ')  --  ' , (": 0{sizes) , ' points'
smoutput '  Cluster 1: (' , (6j2 ": (<1;0){sorted) , ', ' , (6j2 ": (<1;1){sorted) , ')  --  ' , (": 1{sizes) , ' points'
smoutput '  Cluster 2: (' , (6j2 ": (<2;0){sorted) , ', ' , (6j2 ": (<2;1){sorted) , ')  --  ' , (": 2{sizes) , ' points'
smoutput ''

NB. ── 5. K-Nearest Neighbors ────────────────────────────────────────
NB. 20 random reference points in 3D, query a few
smoutput '--- 5. K-Nearest Neighbors ---'
nref =: 20
Xref =: (nref ?@$ 0) ,. (nref ?@$ 0) ,. (nref ?@$ 0)

m =: '' conew 'jknn'
train__m Xref
smoutput 'Indexed ' , (": nref) , ' random reference points in 3D'
smoutput ''

NB. Query the first 3 reference points (should find themselves as nearest)
Xquery =: 3 {. Xref
'neighbors distances' =: 2 search__m Xquery
destroy__m ''

smoutput 'Querying first 3 reference points back (k=2):'
smoutput '  Each should find itself (distance=0) as nearest neighbor.'
smoutput ''
smoutput '  Query  Nearest  Dist     2nd nearest  Dist'
smoutput '  -----  -------  ----     -----------  ----'
smoutput '  pt 0   pt ' , (": (<0;0){neighbors) , '    ' , (5j2 ": (<0;0){distances) , '     pt ' , (": (<0;1){neighbors) , '         ' , 5j2 ": (<0;1){distances
smoutput '  pt 1   pt ' , (": (<1;0){neighbors) , '    ' , (5j2 ": (<1;0){distances) , '     pt ' , (": (<1;1){neighbors) , '         ' , 5j2 ": (<1;1){distances
smoutput '  pt 2   pt ' , (": (<2;0){neighbors) , '    ' , (5j2 ": (<2;0){distances) , '     pt ' , (": (<2;1){neighbors) , '         ' , 5j2 ": (<2;1){distances
smoutput ''

NB. ── 6. PCA ────────────────────────────────────────────────────────
NB. 200 points in 3D that mostly vary along one direction
smoutput '--- 6. PCA ---'
smoutput 'Data: 200 points in 3D along the (1,1,1) direction + tiny noise'
smoutput ''
n =: 200
t =: (n $ 0.05) * i. n
noise1 =: 0.01 * (n ?@$ 0) - 0.5
noise2 =: 0.01 * (n ?@$ 0) - 0.5
NB. Strong signal along (1,1,1) direction, tiny noise in other dims
Xdata =: (t + noise1) ,. (t + noise2) ,. (t + noise1 + noise2)

smoutput 'Original data shape:    ' , (": 0 { $ Xdata) , ' x ' , ": 1 { $ Xdata

m =: 0 conew 'jpca'
'transformed eigvals varret' =: 1 apply__m Xdata
destroy__m ''

smoutput 'Reduced data shape:     ' , (": 0 { $ transformed) , ' x ' , ": 1 { $ transformed
smoutput ''
smoutput 'Variance retained by 1 component: ' , (6j2 ": 100 * varret) , '%'
smoutput '  (>99% expected since data lies along a single direction)'
smoutput ''

smoutput '=== All examples complete ==='

exit 0
