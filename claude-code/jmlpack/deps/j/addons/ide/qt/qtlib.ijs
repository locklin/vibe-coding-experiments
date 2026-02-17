NB.

require 'ide/qt/keys'
require 'ide/qt/gl2'

coclass 'jqtide'
NB. image

NB. Format              Description               Qt's support
NB. BMP    Windows Bitmap                         Read/write
NB. GIF    Graphic Interchange Format (optional)  Read
NB. JPG    Joint Photographic Experts Group       Read/write
NB. JPEG   Joint Photographic Experts Group       Read/write
NB. PNG    Portable Network Graphics              Read/write
NB. PBM    Portable Bitmap                        Read
NB. PGM    Portable Graymap                       Read
NB. PPM    Portable Pixmap                        Read/write
NB. TIFF   Tagged Image File Format               Read/write
NB. XBM    X11 Bitmap                             Read/write
NB. XPM    X11 Pixmap                             Read/write

RGBSEQ=: 1  NB. Qt

NB. =========================================================
NB. reading file images from memory to argb matrix
NB. y raw image data
getimg=: 3 : 0
if. m [[ 'm wh'=. 0 3{ wdgetimg y;(#y);wh=. 2$2-2 do.
  d=. _2 ic memr m,0,(*/wh,4),2
  wdreadimg 2#<<0  NB. dispose temp image in qt side
  (|.wh)$ fliprgb^:(-.RGBSEQ) d
else.
  0 0$2-2
end.
)

NB. =========================================================
NB. reading file images to argb matrix
NB. y image file name
readimg=: 3 : 0
if. m [[ 'm wh'=. 0 2{ wdreadimg (utf8 ,y);wh=. 2$2-2 do.
  d=. _2 ic memr m,0,(*/wh,4),2
  wdreadimg 2#<<0  NB. dispose temp image in qt side
  (|.wh)$ fliprgb^:(-.RGBSEQ) d
else.
  0 0$2-2
end.
)

NB. =========================================================
NB. write image data to memory
NB. x rgb matrix
NB. y 'jpeg|png|ico|bmp' [ ; 'quality' ; '[0,100]|_1' ]
putimg=: 4 : 0
if3=. (3=#$x) *. 3={:$x
if. if3 do.
  x=. setalpha 256&#. x
end.
'h w'=. $x
d=. ,x
type=. 'jpeg'
opt=. ''
quality=. _1
if. 1= #y=. boxxopen y do.
  type=. >@{.y
elseif. 2< #y do.
  type=. >@{.y
  opt=. 2{.}.y
end.
if. 'jpg'-:type do. type=. 'jpeg'
elseif. 'tif'-:type do. type=. 'tiff'
end.
type=. toupper type
if. 'quality'-:>@{.opt do. quality=. <. >@{:opt end.
d=. fliprgb^:(-.RGBSEQ) d
'm len'=. 0 3{ wdputimg (2 ic d); (w,h); (len=. ,_1); type; quality
if. m do.
  z=. memr m,0,len,2
  wdputimg (4#(<<0)),<0  NB. dispose temp bytearry in qt side
  z
else.
  ''
end.
)

NB. =========================================================
NB. write image data to file
NB. x rgb matrix
NB. y filename [ ; 'jpeg|png|ico|bmp' [ ; 'quality' ; [0,100]|_1 ]]
writeimg=: 4 : 0
if3=. (3=#$x) *. 3={:$x
if. if3 do.
  x=. setalpha 256&#. x
end.
'h w'=. $x
d=. ,x
y=. boxopen y
f=. > fboxname {. y
type=. 'jpeg'
opt=. ''
quality=. _1
if. 1= #y do.
  type=. }. (}.~ i:&'.') f
else.
  type=. >1{y
  opt=. 2{.2}.y
end.
type=. tolower type
if. 'jpg'-:type do. type=. 'jpeg'
elseif. 'tif'-:type do. type=. 'tiff'
end.
type=. toupper type
if. 'quality'-:>@{.opt do. quality=. <. >@{:opt end.
d=. fliprgb^:(-.RGBSEQ) d
wdwriteimg (2 ic d); (w,h); f; type; quality
EMPTY
)
NB.

cocurrent IFQT{'jqtide';'z'

NB. =========================================================
wdreadimg=: ('"',libjqt,'" wdreadimg  ',(IFWIN#'+'),' x *c *i')&cd
wdgetimg=: ('"',libjqt,'" wdgetimg  ',(IFWIN#'+'),' x *c i *i')&cd
wdwriteimg=: ('"',libjqt,'" wdwriteimg  ',(IFWIN#'+'),' i *c *i *c *c i')&cd
wdputimg=: ('"',libjqt,'" wdputimg  ',(IFWIN#'+'),' x *c *i *i *c i')&cd

NB. =========================================================
NB. preserved for compatibility
getsha1=: 3 : 0
gethash 'sha1';y
)

NB. =========================================================
gethash=: 3 : 0
't m'=. y
t gethash m
:
m=. ,y
c=. '"',libjqt,'" gethash ',(IFWIN#'+'),' i *c *c i * *i'
'r t m w p n'=. c cd (tolower x);m;(#m);(,2);,0
res=. memr p,0,n
if. r do.
  res (13!:8) 3
end.
res
)
