NB. J qtide

require 'project'

coclass 'jqtide'

3 : 0''
if. IFQT do.
  require 'ide/qt/qtlib'
end.
EMPTY
)
'jws_onOpen jws_onClose jws_onMessage jws_onError jws_onSslError jws_onStateChange'=: i.6
boxj2utf8=: 3 : 0
if. 1 < #$y do. y return. end.
b=. (16+i.11) { a.
if. -. 1 e. b e. y do. y return. end.
y=. ucp y
a=. ucp '┌┬┐├┼┤└┴┘│─'
x=. I. y e. b
utf8 (a {~ b i. x { y) x } y
)
clipread=: wd@('clippaste'"_)
clipwrite=: 3 : 0
txt=. boxj2utf8 flatten ":y
wd 'clipcopy *',txt
#txt
)
dbjqt=: 3 : 0
if. y do.
  if. _1 = 4!:0 <'jdb_open_jdebug_' do.
    0!:0 <jpath '~addons/ide/qt/debugs.ijs'
  end.
  jdb_open_jdebug_''
else.
  jdb_close_jdebug_ :: ] ''
end.
)
flatten=: 3 : 0
dat=. ": y
select. # $ dat
case. 1 do.
case. 2 do.
  }. , LF ,. dat
case. do.
  dat=. 1 1}. _1 _1}. ": < dat
  }: (,|."1 [ 1,.-. *./\"1 |."1 dat=' ')#,dat,.LF
end.
)
gridindex=: 3 : 0
'rws cls sel ndx shp'=. y
d=. (sel,rws,cls) |: i.shp
r=. (sel{shp),(*/rws{shp),*/cls{shp
,(<ndx) { r ($,) d
)
showevents=: 3 : 0
select. {. y,1
case. 0 do.
  4!:55 <'wdhandler_debug_z_'
case. 1 do.
  wdhandler_debug_z_=: 3 : 'smoutput sysevent'
case. 2 do.
  wdhandler_debug_z_=: 3 : 'smoutput wdq'
case. 3 do.
  wdhandler_debug_z_=: 3 : 'if. -. ''_mmove''-:_6{.sysevent do. smoutput sysevent end.'
case. 4 do.
  wdhandler_debug_z_=: 3 : 'if. -. ''_mmove''-:_6{.sysevent do. smoutput wdq end.'
end.
EMPTY
)
textview=: 3 : 0
p=. boxopen y
if. 1<#$p do.
  p=. <":p
elseif. 2=#p do.
  p=. 1 0 1 #^:_1 p
end.
'title caption text'=. _3 {. p
wd 'textview *;',title,';',caption,';',flatten text
)
echo_z_=: wd@'msgs'^:2 [ 1!:2&2
addons_msg=: 0 : 0
The XX are not yet installed.

To install, select menu Tools|Package Manager and install package YY.
)
addons_missing=: 3 : 0
'name addon script'=. y
if. fexist script do. 0 return. end.
sminfo name;addons_msg rplc 'XX';name;'YY';addon
1
)
demoqt=: 3 : 0
p=. jpath '~addons/demos/qtdemo/qtdemo.ijs'
if. addons_missing 'qt demos';'demos/qtdemo';p do. return. end.
load p
)
demowd=: 3 : 0
p=. jpath '~addons/demos/wd/demos.ijs'
if. addons_missing 'Showcase demos';'demos/wd';p do. return. end.
load p
)
labs_run=: 3 : 0
p=. jpath '~addons/labs/labs/lab.ijs'
if. addons_missing 'labs';'labs/labs';p do. return. end.
require p
if. 0 e. $y do.
  require '~addons/labs/labs/labs805.ijs'
  labselect_jlab805_'' return.
end.
if. y -: 1 do. y=. ':' end.
empty lab_jlab_ y
)
coclass 'jqtide'

Qt_Key_Escape                 =: 16b00f800
Qt_Key_Tab                    =: 16b00f801
Qt_Key_Backtab                =: 16b00f802
Qt_Key_Backspace              =: 16b00f803
Qt_Key_Return                 =: 16b00f804
Qt_Key_Enter                  =: 16b00f805
Qt_Key_Insert                 =: 16b00f806
Qt_Key_Delete                 =: 16b00f807
Qt_Key_Pause                  =: 16b00f808
Qt_Key_Print                  =: 16b00f809
Qt_Key_SysReq                 =: 16b00f80a
Qt_Key_Clear                  =: 16b00f80b
Qt_Key_Home                   =: 16b00f810
Qt_Key_End                    =: 16b00f811
Qt_Key_Left                   =: 16b00f812
Qt_Key_Up                     =: 16b00f813
Qt_Key_Right                  =: 16b00f814
Qt_Key_Down                   =: 16b00f815
Qt_Key_PageUp                 =: 16b00f816
Qt_Key_PageDown               =: 16b00f817
Qt_Key_Shift                  =: 16b00f820
Qt_Key_Control                =: 16b00f821
Qt_Key_Meta                   =: 16b00f822
Qt_Key_Alt                    =: 16b00f823
Qt_Key_CapsLock               =: 16b00f824
Qt_Key_NumLock                =: 16b00f825
Qt_Key_ScrollLock             =: 16b00f826
Qt_Key_F1                     =: 16b00f830
Qt_Key_F2                     =: 16b00f831
Qt_Key_F3                     =: 16b00f832
Qt_Key_F4                     =: 16b00f833
Qt_Key_F5                     =: 16b00f834
Qt_Key_F6                     =: 16b00f835
Qt_Key_F7                     =: 16b00f836
Qt_Key_F8                     =: 16b00f837
Qt_Key_F9                     =: 16b00f838
Qt_Key_F10                    =: 16b00f839
Qt_Key_F11                    =: 16b00f83a
Qt_Key_F12                    =: 16b00f83b
helpcontext0=: 3 : 0
require '~addons/ide/qt/help.ijs'
helpcontext y
)

helpcontext1=: 3 : 0
require '~addons/ide/qt/help.ijs'
helpcontext1 y
)
Multiline=: (,each '01234'),;:'noun adverb conjunction verb monad dyad'

ismultiline=: 3 : 0
if. 0=#y do. 0 return. end.
t=. ;: :: 0: y
if. t-:0 do. 0 return. end.
if. (<'Note') = {.t do. 1 return. end.
n=. I. }: t e. Multiline
if. 0=#n do. 0 return. end.
if. (<'define') e. (n+1){t do. 1 return. end.
(,each ':0') e. (n+/1 2){t,<''
)
dirdef_begin=: 3 : 0
t=. ;: :: 0: y
if. t-:0 do. return. end.
b=. t = <'{{'
e=. (t = <'}}') *. +./\ b
0 >. +/ b - e
)
dirdef_end=: 3 : 0
t=. ;: :: 0: y
if. t-:0 do. return. end.
b=. t = <'{{'
e=. t = <'}}'
+/ e - b
)
getnextentry=: 3 : 0
p=. 1
s=. 0 pick y
t=. }.y
if. n=. dirdef_begin s do.
  while. #t do.
    p=. p + 1
    s=. s,LF,w=. 0 pick t
    n=. n - dirdef_end w
    if. n <: 0 do. break. end.
    t=. }.t
  end.
end.
if. ismultiline s do.
  n=. 1 + t i. <,')'
  s=. s,LF,tolist n {. t
  p=. p + n
end.
p;s
)
cocurrent IFQT{'jqtide';'z'
wd=: 3 : 0"1
'r c l p n'=. wd1 (,y);(#,y);(,2);(,0)
select. r
case. 0 do.
  EMPTY
case. _1 do.
  memr p,0,n
case. _2 do.
  _2 [\ <;._2 memr p,0,n
case. do.
  (LF,~wd ::(''"_)'qer') (13!:8) 3
end.
)

wd1=: ('"',libjqt,'" wd ', (IFWIN#'+'),' i *c i *x *i') & cd f.
wdhandler_0=: 3 : 0
wdq=: wd 'q'
wd_val=. {:"1 wdq
({."1 wdq)=: wd_val
if. 3=4!:0<'wdhandler_debug' do.
  try. wdhandler_debug'' catch. end.
end.
wd_ndx=. 1 i.~ 3 = 4!:0 [ 3 {. wd_val
if. 3 > wd_ndx do.
  wd_fn=. > wd_ndx { wd_val
  if. 13!:17'' do.
    wd_fn~''
  else.
    try. wd_fn~''
    catch.
      wd_err=. 13!:12''
      if. 0=4!:0 <'ERM_j_' do.
        wd_erm=. ERM_j_
        ERM_j_=: ''
        if. wd_erm -: wd_err do. i.0 0 return. end.
      end.
      wd_err=. LF,,LF,.(}.^:('|'e.~{.));._2 ,&LF^:(LF~:{:) wd_err
      wdinfo 'wdhandler';'error in: ',wd_fn,wd_err
    end.
  end.
end.
i.0 0
)
wdhandler=: 3 : 0
wdq=: wd 'q'
wd_val=. {:"1 wdq
({."1 wdq)=: wd_val
if. 3=4!:0<'wdhandler_debug' do.
  try. wdhandler_debug'' catch. end.
end.
wd_ndx=. 1 i.~ 3 = 4!:0 [ 3 {. wd_val
if. 3 > wd_ndx do.
  wd_fn=. > wd_ndx { wd_val
  if. 13!:17'' do.
    wd_fn~''
  else.
    try. wd_fn~''
    catch.
      wd_err=. 13!:12''
      if. 0=4!:0 <'ERM_j_' do.
        wd_erm=. ERM_j_
        ERM_j_=: ''
        if. wd_erm -: wd_err do. i.0 0 return. end.
      end.
      wd_err=. LF,,LF,.(}.^:('|'e.~{.));._2 ,&LF^:(LF~:{:) wd_err
      wdinfo 'wdhandler';'error in: ',wd_fn,wd_err
    end.
  i.0 0
  end.
else. i. 0 0
end.
)
wdhandlerx=: 3 : 0
loc=. <,y
if. 0 <: 18!:0 loc do.
  wdhandler__loc''
else.
  msg=. 'no locale for event handler: ',>loc
  if. #wdq=. wd 'q' do.
    msg=. msg, ', event: ',(<1 1) pick wdq
  end.
  smoutput msg
end.
)
wdclippaste=: (wd bind 'clippaste') :: (''"_)
wdqq=: (wd bind 'q') :: (''"_)
wdqchildxywh=: (0 ". [: wd 'qchildxywh ' , ]) :: (0 0 0 0"_)
wdqcolor=: (0 ". [: wd 'qcolor ' , ":) :: ( 0 0 0"_)
wdqd=: (wd bind 'qd') :: (''"_)
wdqer=: (wd bind 'qer') :: (''"_)
wdqform=: (0 ". wd bind 'qform') :: (0 0 800 600"_)
wdqhinst=: (0 ". wd bind 'qhinst') :: 0:
wdqhwndc=: (0 ". [: wd 'qhwndc ' , ]) :: 0:
wdqhwndp=: (0 ". wd bind 'qhwndp') :: 0:
wdqhwndx=: (0 ". wd bind 'qhwndx') :: 0:
wdqm=: (0 ". wd bind 'qm') :: (800 600 8 16 1 1 3 3 4 4 19 19 0 0 800 570"_)
wdqp=: (wd bind 'qp') :: (''"_)
wdqprinters=: (wd bind 'qprinters') :: (''"_)
wdqpx=: (wd bind 'qpx') :: (''"_)
wdqscreen=: (0 ". wd bind 'qscreen') :: (264 211 800 600 96 96 32 1 _1 36 36 51"_)
wdqwd=: (wd bind 'qwd') :: ('jqt'"_)
wdcenter=: 0:
wdfit=: 0:

wdreset=: wd bind 'reset'
wdforms=: <;._2;._2 @ wd bind 'qpx'
wdisparent=: (boxopen e. 0: {"1 wdforms) ::0:
wdishandle=: (boxopen e. 1: {"1 wdforms) ::0:
wdinfo=: 3 : 0
'a b'=. _2{. boxopen y
if. 2=#$b=. ":b do. b=. }.,LF,.b end.
f=. 8 u: DEL&, @ (,&DEL) @ -.&(0 127{a.)
empty wd 'mb info ',(f a),' ',(f b)
)
wdquery=: 3 : 0
0 3 wdquery y
:
msg=. ' mb_'&,&.> res=. ;:'ok cancel yes no save discard'
t=. x [ 'a b'=. _2{. boxopen y
if. 2=#$b=. ":b do. b=. }.,LF,.b end.
f=. 8 u: DEL&, @ (,&DEL) @ -.&(0 127{a.)
m=. 'mb query', (;t{msg), ' ', (f a),' ',(f b)
res i. <wd m
)
mbopen=: 3 : 0
jpathsep wd 8 u: 'mb open1 ',y
)
mbsave=: 3 : 0
jpathsep wd 8 u: 'mb save ',y
)
wdget=: 4 : 0
nms=. {."1 y
vls=. {:"1 y
if. L. x do. vls {~ nms i. ,&.>x
else. > vls {~ nms i. <,x
end.
)

wdpclose=: [: wd :: empty 'psel ' , ';pclose' ,~ ":
3 : 0^:(IFQT > IFJHS)''
dirmatch=: 3 : 'wd ''dirmatch '', ; dquote&.> 2 {. boxopen y'
open=: 3 : 'wd ''openj *'' , > {. getscripts_j_ y'
immexj=: [: wd 'immexj *'&,
smact=: wd bind 'smact'

clipread=: clipread_jqtide_
clipwrite=: clipwrite_jqtide_
textview=: textview_jqtide_
EMPTY
)
coclass 'jqtide'
JQTREQ=: '1.8.1'
JQTMSG=: 0 : 0
The JQt binary needs updating.

To do so, close this session, then load jconsole and run:

   getqtbin''
or
   getqtbin'slim'

or run XX in the J installation folder.
)
checkjqtversion=: 3 : 0
f=. 1000 #. 0 ". ' ' I.@('.'=])} ]
ver=. wd 'version'
ver=. (ver i.'(') {. ver
JQTVER=: f (1+ver i.'/') }. ver
act=. f (<./ ver i.'/s') {. ver
req=. f JQTREQ
if. req <: act do. return. end.
msg=. JQTMSG rplc 'XX';'updatejqt.',IFWIN pick 'sh';'cmd'
sminfo 'JQt';msg
)
checkjqtversion^:IFQT''

cocurrent 'base'
