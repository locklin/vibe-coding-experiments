coclass 'jdebug'
jdb_deb=: #~ (+. (1: |. (> </\)))@(' '&~:)
jdb_dlb=: }.~ =&' ' i. 0:
jdb_dtb=: #~ ([: +./\. ' '&~:)
jdb_isexplicit=: (<,':')"_ -: {.@>@(5!:1)@<
jdb_issparse=: 512&[ < 3!:0
jdb_splitind=: ('__'&E. i. 1:) ({. ; 2: }. }.) ]
jdb_takeafter=: [: ] (#@[ + E. i. 1:) }. ]
jdb_taketo=: [: ] (E. i. 1:) {. ]
jdb_tolist=: }.@;@:((10{a.)&,@,@":&.>)
jdb_toDEL=: [: ; ' "'&, @ (,&'"') &.>

jdb_boxopen=: boxopen f.
jdb_boxxopen=: boxxopen f.
jdb_empty=: empty f.
jdb_expand=: expand f.
jdb_isparent=: 0:`wdisparent@.IFQT f.
jdb_datatype=: datatype f.
jdb_getactive=: jdb_wd bind 'sm get active'
jdb_setactive=: jdb_wd @ ('sm focus '&,)
jdb_smact=: jdb_setactive @ jdb_getactive
jdb_smoutput=: 0 0&$@(1!:2&2)
jdb_val=: ]`($.^:_1) @. jdb_issparse
jdb_wdforms=: <;._2;._2@jdb_wd@('qpx'"_)
jdb_wd1=: 0:`wd1@.IFQT f.
jdb_dbs=: 13!:1
jdb_dbsq=: 13!:2
jdb_dbss=: 13!:3
jdb_dbsig=: 13!:8
jdb_dberm=: 13!:12
jdb_dbstk=: 13!:13
jdb_dbtrace=: 13!:16

jdb_imxq=: 9!:26
jdb_imxs=: 9!:27
jdb_imxsq=: 9!:28
jdb_imxss=: 9!:29
NAMECHRS=: '_',a.{~(,(i.10)+a.i.'0'),,(i.26) +/ a.i.'Aa'
jdb_getleftsepndx=: 0: i.~ e. & NAMECHRS
jdb_getrightsepndx=: 0: i.~ e. & (NAMECHRS,'.:')
jdb_codelines=: [: I. ('NB.'&-: @ (3&{.) +: [: *./ ' '&=)&>
jdb_cutopen=: 3 : 0
y jdb_cutopen~ (' ',LF) {~ LF e. ,y
:
if. L. y do. y return. end.
if. 1 < #$y do. <"_1 y return. end.
(<'') -.~ (y e.x) <;._2 y=. y,1{.x
)
jdb_default=: 4 : 0
if. _1 = 4!:0 <x do.
  ".x,'=: y'
end.
empty''
)
jdb_edit=: 3 : 0
jdb_lxsoff''
jdb_stopread''
jdb_dbss''
if. 0 -: edit_z_ :: 0: y do.
  jdb_info 'Unable to open definition of ',":>y
end.
jdb_stopwrite''
jdb_lxson''
)
jdb_flatten=: 3 : 0
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
jdb_fullname=: 3 : 0
LOCALE jdb_fullname y
:
if. 2=#jdb_boxopen y do. y return. end.

x=. > x
y=. > y

if. '_' = {: y do.
  ndx=. (}:y) i: '_'
  j=. (ndx+1) }. }: y
  loc=. j, (0=#j) # 'base'
  (ndx {. y) ; loc
else.
  y ; x
end.
)
jdb_getcursorline=: 3 : '+/ LF = ({. 0 ". lines_select) {. lines'
jdb_getdefs=: (#~ _2: < 4!:0) @ ;:
jdb_getdefs=: 3 : 0
y=. ;: y
if. 1 e. msk=. (1: e. '__'&E.) &> y do.
  bal=. (-.msk) # y
  msk=. (_2 < 4!:0 bal) (I. -.msk)} msk
  msk # y
else.
  y #~ _2 < 4!:0 y
end.
)
jdb_getnameat=: 4 : 0

txt=. y
sel=. x

'px py'=. sel

rtx=. |. px{.txt
rnd=. jdb_getleftsepndx rtx
beg=. |. rnd {. rtx
bps=. px - rnd
mid=. (py-px) {. px }. txt
etx=. py }. txt

end=. jdb_getrightsepndx etx
bit=. beg, mid, end {. etx

if. 0=#bit-.' ' do.
  name=. ''
else.
  wds=. ;:bit
  len=. # &> }:wds
  ndx=. 0 i.~ (#beg) >: +/\ len
  name=. ndx >@{ wds
  off=. +/ ; ndx {. len
  fnd=. 1 i.~ name E. off }. bit
  sel=. bps+off+fnd
end.

name=. jdb_validname name

if. #name do.
  hit=. {.sel
  ndx=. hit + 0, #name
  ndx;name
else.
  '';''
end.
)
jdb_getnamesat=: 4 : 0
'bgn end'=. x
txt=. bgn }. end {. y
nms=. jdb_getdefs txt
nms=. nms -. MNUVXY
if. #nms do.
  bgn=. bgn + 2 + txt i. ']'
  (bgn,end) ; < nms
else.
  '';''
end.
)
jdb_getstack=: (}.~ ('*' i.~ [: > 8 {"1 ]))@(13!:13)
jdb_getwincolor=: 3 : 0
0 0 0
)
jdb_indices=: '['"_ , ": , '] '"_
jdb_indexit=: ([: jdb_indices &.> [: i. #) ,&.> ]
jdb_info=: 3 : 0
'a b'=. _2 {. jdb_boxopen y
if. 0 = #a do. a=. 'Debug' end.
if. 2=#$b=. ":b do. b=. }.,LF,.b end.
f=. DEL&, @ (,&DEL) @ -.&(0 127{a.)
empty jdb_wd 'mb info ',(f a),' ',(f b)
)
jdb_isgui=: 3 : 0
if. 0 e. $HWNDP do. 0 return. end.
f=. jdb_wdforms''
if. 0 e. $f do. 0 return. end.
(<HWNDP) e. 1{"1 f
)
jdb_listboxed=: }. @; @: (LF&, &.>)
jdb_listmatrix=: [: }. [: , LF&,.
jdb_lxsoff=: 13!:15 @ (''"_)

cocurrent 'jdebugnopath'
copath ''
z458095869 =: (,: <@(4!:0)) @ ((<'z458095869') -.~ 4!:1@i.@4:)
cocurrent 'jdebug'
jdb_lxson=: 3 : 0
13!:15 '(z458095869_jdebugnopath_$0) jdb_debug_jdebug_ (coname$0)'
if. 0 = 13!:17'' do. 13!:0 [ 1 end.
i. 0 0
)
jdb_minsize=: 3 : 0
a=. 0 ". jdb_wd 'qform'
a=. a >. 0 0,MINWIDTH,MINHEIGHT
jdb_wd 'pmove ', ": a
)
jdb_pack=: [: /:~ [: (, ,&< ".) &> ;: ::]
jdb_nextline=: 3 : 0
if. 0 e. #CODELINES do. 0 return. end.
if. y=0 do.
  ((i.<./) |CODELINES - MOVELINE) { CODELINES
else.
  ndx=. CODELINES i. MOVELINE
  ndx=. 0 >. (_1 + #CODELINES) <. ndx + y
  ndx { CODELINES
end.
)
jdb_ppget=: 3 : 0
h=. jdb_wd :: ] 'qhwndp'
jdb_empty HWNDPX=: (-. h-:HWNDP)#h
)
jdb_ppset=: 3 : 0
if. #HWNDPX do. jdb_wd :: ] 'psel ', HWNDPX end.
jdb_empty''
)
jdb_query=: 3 : 0
0 3 wdquery y
:
msg=. ' mb_'&,&.> res=. ;:'ok cancel yes no save discard'
t=. x [ 'a b'=. _2{. boxopen y
if. 2=#$b=. ":b do. b=. }.,LF,.b end.
f=. 8 u: DEL&, @ (,&DEL) @ -.&(0 127{a.)
m=. 'mb query', (;t{msg), ' ', (f a),' ',(f b)
res i. <jdb_wd m
)
jdb_selstack=: 3 : 0
a=. 2 }. jdb_dbstk''
if. 0=#a do. return. end.
b=. ({."1 a) i. jdb_boxopen y
if. b=#a do. return. end.
b { a
)
jdb_shortname=: 3 : 0
if. '_' = {: y do.
  y {.~ (}:y) i: '_'
else.
  y
end.
)
jdb_validname=: 3 : 0
if. 0=#y do. '' return. end.
if. 1 e. b=. '__' E. }: y do.
  if. 1 ~: +/b do.
    '' return.
  else.
    ndx=. I. b
  end.
  if. _2 e. 4!:0 (ndx{.y);(ndx+2)}.y do.
    ''
  else.
    y
  end.
else.
  if. _2 = 4!:0 <y do.
    ''
  else.
    y
  end.
end.
)
jdb_wd=: 3 : 0"1
'r c l p n'=. jdb_wd1 (,y);(#,y);(,2);(,0)
select. r
case. 0 do.
  EMPTY
case. _1 do.
  (15!:1) p,0,n
case. _2 do.
  _2 [\ <;._2 (15!:1) p,0,n
case. do.
  (jdb_wd ::(''"_) 'qer') (13!:8) 3
end.
)
ERM_j_=: ''

ERRORS=: '';(9!:8''),<'Unknown Error'
ERRORCODES=: (i.#ERRORS) -. 0 18

IFDISSECT=: 'true' -: 0 1&{::@:wd ::0: 'qtstate debugdissect'
MINWIDTH=: 540
MINHEIGHT=: 500

MNUV=: ;: 'm n u v m. n. u. v.'
MNUVXY=: ;: 'm n u v x y m. n. u. v. x. y.'
NVX=: ;: 'n v x n. v. x.'

PTOP=: 1
NULL=: $0
STACKPOS=: 0
SMBOTH=: 0
STOPNONE=: '';0;0;NULL;NULL
SHOWWID=: 80
jdb_vSHOWWID=: SHOWWID"_

TABCURRENT=: ''
TYPES=: 'acv'

STOPNONE=: '';0;0;NULL;NULL
TYPES=: 'acv'
jdb_debuginit=: 3 : 0
jdb_stopwrite STOPS=: i.0 5
DISSECTOPTIONS=:0 5$a:
'HWNDP' jdb_default ''
if. 0>4!:0 <'WINPOS' do.
  WINPOS=: 0 ". (<0 1) >@{ jdb_wd 'qtstate debugpos'
end.
jdb_debugreset''
)
jdb_debugreset=: 3 : 0
CODELINES=: ''
CURRENTLINE=: 0
ERRMSG=: ''
ERRNUM=: 0
GDEFS=: ''
HWNDPX=: ''
LDEFS=: ''
LOCALE=: <'base'
LOCALNAMES=: ''
LOCALTYPES=: ''
LOCALVALS=: ''
LXS=: ''
MOVELINE=: 0
NAME=: ''
NUMLINES=: 0
SCROLL=: 0
STACK=: i.0 9
SMNAMES=: i.0 2
SMLOCALE=: ''
SSTACKS=: ''
SSNAMES=: ''
STACKLOCALS=: ''
STATE=: 0
STOPLAST=: ''
WATCH=: ''
)
jdb_restore=: 3 : 0
jdb_ppset''
jdb_lxson ''
xsstg =. 'jdb_imxhandler_jdebug_ 1'
if. #y do.
  'line dissectopts' =. y
  xsstg =. ('((', (5!:5 <'dissectopts') , ') dissect ',(jdb_quote line), ') ') , xsstg
end.
jdb_imxs xsstg
jdb_imxss 1
i. 0 0
)
jdb_imxhandler=: 3 : 0
if. 1 >: # 13!:13'' do. jdb_clear'' end.
empty''
:
if. '' -: $x do. autodissectlocale =: x
elseif. *#x do. wdinfo 'Dissect message';x
end.
jdb_imxhandler 1
)
jdb_destroyad =: 3 : 0
if. #autodissectlocale do.
  if. autodissectlocale e. 18!:1 (1) do. destroy__autodissectlocale '' end.
end.
autodissectlocale =: 0$a:
empty''
)
jdb_inactive=: 3 : '0 e. #NAME'
jdb_clear=: 3 : 0
hx=. HWNDPX
jdb_debugreset''
if. jdb_isparent <'jdebug' do.
  jdb_destroyad''
  jdb_wd 'psel jdebug'
  jdb_swap 'jdbnone'
end.
if. #hx do.
  jdb_wd :: jdb_smact 'psel ',hx
else.
  jdb_smact ''
end.
13!:0 [ 1
jdb_lxson''
)
jdb_close=: 3 : 0
if. jdb_isgui'' do.
  jdb_wd 'psel ',HWNDP
  WINPOS=: 0 ". jdb_wd 'qform'
  jdb_wd 'pclose'
end.
HWNDP=: ''
TABCURRENT=: ''
jdb_lxsoff ''
jdb_imxss 0
jdb_imxs ''
jdb_destroyad''
jdb_debuginit''
13!:0 [ 0
)
jdb_open=: 3 : 0
forcereopen =. {.!.0 y
a=. jdb_getactive''
jdb_debuginit''
ERM_j_=: ''
if. (-.forcereopen)*#jdb_getstack'' do.
  jdb_debug ''
  jdb_setactive a
  ACTIVE=: ''
else.
  jdb_ppget 0
  jdebug_run 0
  jdb_restore''
  ACTIVE=: a
end.
)
j=. 0 : 0
Enter          !single step over
F5~~~~~~~~     !run
Ctrl+Shift+F5  !run from next line
F6~~~~~~~~     !single step into
F7~~~~~~~~     !single step over
F8~~~~~~~~     !step out of current definition
F9~~~~~~~~     !toggle stop on cursor line
Ctrl+Shift+F9  !remove all stops
Ctrl+T         !toggle topmost attribute
Ctrl+W         !write current line to session
)

SHORTCUTS=: ' ' (I. j='~') } TAB (I. j='!')} j
jdb_swapfkey=: 3 : 0
jdebug_f5_fkey=: ]
jdebug_f6_fkey=: ]
jdebug_f7_fkey=: ]
jdebug_f8_fkey=: ]
jdebug_f9_fkey=: ]
jdebug_f5ctrlshift_fkey=: ]
jdebug_f9ctrlshift_fkey=: ]
if. TABCURRENT -: 'jdbmain' do.
  jdebug_f5_fkey=: jdebug_run_button
  jdebug_f6_fkey=: jdebug_stepinto_button
  jdebug_f7_fkey=: jdebug_stepover_button
  jdebug_f8_fkey=: jdebug_stepout_button
  jdebug_f9_fkey=: jdbmain_stopline_button
  jdebug_f5ctrlshift_fkey=: jdebug_runnext
  jdebug_f9ctrlshift_fkey=: jdebug_clearstops
elseif. TABCURRENT -: 'jdbstop' do.
  jdebug_f9_fkey=: jdbstop_stopline_button
end.
0
)
jdb_debug=: 3 : 0
jdb_lxsoff''

if. -. jdb_isgui'' do.
  13!:15''
  13!:0[0
  HWNDP=: TABCURRENT=: ''
  return.
end.

stack=. jdb_getstack''
if. 0 e. #stack do.
  jdb_lxson'' return.
end.
stack=. {. stack

jdb_ppget 0

if. #y do.
  LOCALE=: y
else.
  LOCALE =: <'base'
end.
ERM_j_=: jdb_dberm''
'NAME ERRNUM CURRENTLINE'=: 3 {. stack
MOVELINE=: CURRENTLINE
MOVELINES=: ,MOVELINE
ERRMSG=: (ERRNUM <. <:#ERRORS) >@{ ERRORS
jdb_lexwin ''
if. (*#y) *. IFDISSECT *. AUTODISSECT do.

  jdb_destroyad''
  jdebug_dissectcurrent_run CURRENTLINE
else. jdb_restore ''
end.
:
LOCALTYPES =: x
jdb_debug y
)
jdb_lexwin=: 3 : 0
if. 0 e. #NAME do. '' return. end.
jdb_stopread''
STACK=: jdb_getstack''
jdb_lexwin1 ''
i. 0 0
)
jdb_lexwin1=: 3 : 0
j=. (MOVELINE ; ERRMSG) jdb_stackrep STACK

if. 0 = L. j do. return. end.

'lines stack values'=. j

mrg=. '>' CURRENTLINE} NUMLINES # ' '

if. MOVELINE ~: CURRENTLINE do.
  mrg=. ('-+' {~ MOVELINE > CURRENTLINE) MOVELINE} mrg
end.

stp=. jdb_stopget''
lines=. (<"1 stp,.mrg) ,&.> lines
lines=. jdb_dtb &.> lines

jdebug_run 1

SCROLL=: 0 >. (NUMLINES - 6) <. MOVELINE - 2

1 jdb_writelines lines
jdb_wd 'set stack text *',jdb_listboxed stack
jdb_wd 'set value text *',jdb_listboxed values
jdb_wd 'pactive'
)
EX2=: '1234' ;&,&> ':'
EX0=: EX2 ,. < ,'0'
EX1=: EX2 ,. < ,'('
jdb_boxrep=: 4 : 0

'tac nmc'=. x
if. tac do.
  rep=. SUBTC (I.y=LF) } y
  nmc;2$<<rep return.
end.
hdr=. ;: LF jdb_taketo y
if. 1 e. , b=. EX0 E."1 hdr do.
  cls=. >: (+./"1 b) i. 1
  rep=. }. }: <;._2 y,LF
elseif. 1 e. , b=. EX1 E."1 hdr do.
  cls=. >: (+./"1 b) i. 1
  bgn=. 3 + 1 i.~ +./ b
  hdr=. bgn }. hdr
  hdr=. ; (hdr i. <,')') {. hdr
  try.
    rep=. ". hdr
  catch.
    rep=. hdr
  end.
elseif. 1 e. , b=. EX2 E."1 hdr do.
  cls=. >: (+./"1 b) i. 1
  ndx=. 2 + 1 i.~ +./ b
  try.
    rep=. ". ndx >@{ hdr
  catch.
    rep=. }. }: ndx >@{ hdr
  end.
elseif. do.
  cls=. _1
  rep=. y
end.
rep=. jdb_boxxopen rep
ind=. rep i. < ,':'
if. ind < #rep do.
  cls ; (ind {. rep) ; < (1+ind) }. rep
else.
  cls ; rep ; < rep
end.
)
j=. SHOWWID & {.
jdb_curtail=: ]`(j f.) @. (jdb_vSHOWWID < #)
j=. (SHOWWID-3) & {. , '...'"_
jdb_curtailed=: ]`(j f.) @. (jdb_vSHOWWID < #)
jdb_getdrep=: 3 : 0
'name loc'=. y

if. 0 e. $name do. '' return. end.

bloc=. <loc

if. 1 e. '__' E. }: name do.
  j=. jdb_splitind name
  if. 2 ~: #j do. '' return. end.
  'a b'=. j
  if. 0 ~: nc <b do. '' return. end.
  c=. (b,'_',loc,'_')~
  if. 1 ~: L. c do. '' return. end.
  18!:4 c
  nc=. 4!:0 :: _2: <a
  18!:4 <'jdebug'
  if. -. nc e. 1 2 3 do. '' return. end.
else.
  18!:4 bloc
  nc=. 4!:0 :: _2: <name
  18!:4 <'jdebug'
  if. -. nc e. 1 2 3 do. '' return. end.
end.

18!:4 bloc
def=. 5!:5 <name
18!:4 <'jdebug'

def
)
jdb_fixlocal=: 4 : 0
if. 0=#x do. '' return. end.
if. 0 = #LOCALTYPES do.
  jdb_fixnoun y
else.
  class =. (0 ,~  > {: LOCALTYPES) {~ ({. LOCALTYPES) i. <x
  if. 0 = class do. jdb_fixnoun y
  else. class jdb_fixdef y
  end.
end.
)
jdb_fixdef=: 4 : 0
type=. 'acvu' {~ 1 2 3 i. x
type,' ',jdb_subchars y
)
jdb_fixnoun=: 3 : 0

shape=. $y
len=. */shape

if. 0 = L. y do.
  y=. jdb_fixopen y

else.

  if. 1 = L. y do.
    if. 1 = len do.
      y=. '<',jdb_fixopen >{.y
    else.
      y=. jdb_curtail ,y
      ben=. */ @ $ S: 0 y
      if. 1000 < +/ ben do.
        y=. '{boxed}'
      else.
        y=. }. ; (',(<'&, @ (,&')')) &.> jdb_fixopen &.> y
      end.
    end.
  else.
    y=. '{boxed}'
  end.
  y=. (jdb_repshape shape), y

end.

r=. jdb_curtailed 'n ',jdb_subchars y

)
jdb_fixopen=: 3 : 0

shape=. $y
len=. */shape
rsh=. jdb_repshape shape

if. jdb_issparse y do.
  if. 1000 < len do.
    rsh, '{sparse}' return.
  else.
    t=. $. ^:_1 y
  end.
else.
  t=. y
end.

if. 0=#shape do.
  ":t
else.
  t=. jdb_curtail ,t
  jdb_curtailed rsh,jdb_quoteme t
end.
)
jdb_leaves=: ([: ; <@":@, S: 0) ^: (L. > 0:)
a=. ''''
jdb_quote=: (a&,@(,&a))@ (#~ >:@(=&a))
jdb_quoteme=: ":`jdb_quote @. (2&=@(3!:0))
jdb_repshape=: (0: < #) # ": , '$'"_
SUBTC=: 1{a.
jdb_subtc=: SUBTC & (I. @ (e.&(9 10 12 13{a.)) @] })
jdb_remzero=: -. & ({.a.)
jdb_subchars=: jdb_remzero @: jdb_subtc
jdb_viewname=: 4 : 0
max=. 50000
shape=. $y
scalar=. 0=#shape
type=. jdb_datatype y

dat=. jdb_flatten y
if. max < #dat do.
  dat=. (max {. dat),LF,'...'
end.

tag=. type
tag=. tag, scalar >@{ (' shape ',":shape);' scalar'

if. type -: 'boxed' do.
  tag=. tag, ' depth ',":L. y
end.

dat=. x,LF,LF,tag,LF,LF,dat
s=. 1{a.
jdb_wd 'textview *',s,'view',s,s,dat
)
TIDS=: ' ' ,. 'nacvd' ,. ' '
jdb_addlocale=: 4 : 0
if. ('_' ~: _1 {. y) > 1 e. '__' E. y do.
  if. '_' ~: _1 {. x do.
    t=. '_',(>x),'_'
  else.
    t=. x
  end.
  y,t
else.
  y
end.
)
jdb_showglobals=: 4 : 0
if. 0 e. $y do. '' return. end.

ind=. (1: e. '__'&E.) @: }: &> y
18!:4 x
nmc=. 4!:0 :: _2: "0 y
18!:4 <'jdebug'
nmc=. nmc - ind *. nmc=_1

msk=. nmc = 0
val=. msk # y
med=. nmc > 0
def=. med # y
GDEFS=: GDEFS, def

18!:4 x
if. #val do. val=. jdb_val_jdebug_@". &.> val end.
if. #def do. def=. <@(5!:5) "0 def end.
18!:4 <'jdebug'

val=. jdb_fixnoun &.> val
def=. (med # nmc) jdb_fixdef &.> def

res=. (#msk) # <'{undefined}'
res=. (<'{unknown}') (I. nmc=_2)} res
res=. val (I. msk)} res
res=. def (I. med)} res
)
jdb_stackrep=: 3 : 0
'' jdb_stackrep y
:

if. 0 = #y do. y=. jdb_getstack'' end.
if. 0 = #y do. '' return. end.
LOCALVALS=: 7 {"1 y
y=. 7 {."1 y

STACKLOCALS=: {."1 &.> LOCALVALS
LOCALVALS=: {:"1 > {. LOCALVALS
LOCALNAMES=: > {. STACKLOCALS

'NAME ERRNUM CURRENTLINE'=: 3 {. {. y
if. 0 e. #NAME do. '' return. end.

LDEFS=. GDEFS=. ''
if. #x do.
  'linenum errmsg'=. x
else.
  linenum=. CURRENTLINE
  errmsg=. (ERRNUM <. <:#ERRORS) >@{ ERRORS
end.
nms=. 0{"1 y
lns=. linenum , }. ; 2{"1 y
nmc=. ; 3{"1 y
rps=. 4{"1 y
arglen=. # &> 6{"1 y
val=. (#nmc) # _1
if. 1 e. b=. nmc=3 do.
  val=. (<: b # arglen) (I. b)} val
end.
if. 1 e. b=. (nmc~:3) *. (1: e. MNUV&e.) &> STACKLOCALS do.
  sel=. (<'x')&e. &> b # STACKLOCALS
  val=. sel (I. b)} val
end.
nmc=. nmc + (nmc=3) *. val=1
tac=. 0 = # &> STACKLOCALS
brp=. (>: 0 >. val) >@{ &.> (<"1 tac,.nmc) jdb_boxrep &.> rps
bln=. # &> brp

if. 0=#brp do.
  jdb_info 'Unable to display Debug stack'
  '' return.
end.

if. 0={.bln do.
  jdb_info 'Unable to display top of stack'
  '' return.
end.

if. 0 e. bln do.
  brp=. (<'display not available') (I. bln)} brp
end.
'nam lnm typ'=. (<0;0 2 3) { y
dep=. 0 >@{ brp
LINES=: jdb_dtb &.> dep

dax=. jdb_indices &.> i.#dep
lines=. dax ,&.> dep

NUMLINES=: #lines
CODELINES=: jdb_codelines dep
VALENCE=: {. val
NMC=: {.nmc
lns=. lns <. bln
exl=. ;lns ({ ,& (<'{unknown}')) &.> brp
tacitlines =. -. +./ (LF;'1 :';'2 :';'3 :';'4 :') +./@:E.&>/ rps
headerlines =. tacitlines *. DEBUGNAMESUFFIX&(+./@:E. ,)@> exl
exl =. headerlines DEBUGNAMESUFFIX&(taketo , takeafter)&.> exl
ind=. jdb_indices&.> (tacitlines+headerlines) ({ ;&('tacit';'header'))"0 lns
nmd =. ((-#DEBUGNAMESUFFIX) * (<DEBUGNAMESUFFIX) = (-#DEBUGNAMESUFFIX)&{.&.> nms) }.&.> nms
stack=. nmd ,&.> ind ,&.> exl

st0=. errmsg
if. ERRNUM e. ERRORCODES do.
  erm=. <;._2 ERM_j_
  if. (3=#erm) *. linenum=CURRENTLINE do.
    st0=. st0,LF,NAME,'[',(":linenum),'] ',jdb_dlb }.1 >@{ erm
    stack=. }.stack
  end.
end.

stack=. st0 ; stack
wat=. jdb_cutopen WATCH
dfs=. jdb_getdefs 0 >@{ exl
dfs=. dfs -. MNUVXY
dfs=. ~. wat, ((MNUVXY e. LOCALNAMES)#MNUVXY), dfs
ndx=. LOCALNAMES i. dfs
msk=. ndx = #LOCALNAMES
lcs=. LOCALNAMES jdb_fixlocal &.>&(((-.msk)#ndx)&{) LOCALVALS
glb=. LOCALE jdb_showglobals msk#dfs
vls=. (/:/:msk) { lcs,glb
values=. dfs ,&.> ' ' ,&.> vls
lines;stack;<values
)
a=. , ';'"_ -. {:
b=. i.&':' ({. ; }.@}.) ]
c=. i.&' ' ({. ; b @ }.@}.) ]
d=. c @ jdb_dlb
jdb_stopcut=: ([: d ;._2 a) f.
jdb_stopget=: 3 : 0
jdb_stopgetone NAME;'';VALENCE;NUMLINES;CODELINES
)
jdb_stopgetone=: 3 : 0
'name dummy valence numlines codelines'=. y
'astop ustop'=. jdb_stopson name;valence;codelines
astop=. (astop < numlines) # astop
ustop=. (ustop < numlines) # ustop
'*' ustop} '|' astop} numlines # ' '
)
jdb_stoponall=: 4 : 0
nam=. {. jdb_boxxopen y
jdb_stopread ''
sel=. x
if. sel=2 do.
  sel=. -. (nam,1;1) e. 3 {."1 STOPS
end.
STOPS=: STOPS #~ nam ~: {."1 STOPS
if. sel do.
  STOPS=: STOPS, nam,1;1;'';''
end.
jdb_stopwrite ''
)
jdb_stopread=: 3 : 0
sq=. 13!:2 ''
if. sq -: STOPLAST do. STOPS return. end.
if. 0 = #sq do.
  STOPS=: 0#STOPS
  return.
end.
stp=. /:~ jdb_stopcut sq
nms=. ~. {."1 stp
res=. nms ,"0 1 }. STOPNONE
for_i. 1 2 do.
  col=. i {"1 stp
  msk=. col = <,'*'
  if. 1 e. msk do.
    ndx=. ~. nms i. msk # nms
    res=. (<1) (<ndx;i) } res
  end.
  if. 0 e. msk do.
    mon=. 0 ". &.> (-.msk) # col
    bal=. (-.msk) # nms
    ndx=. nms i. ~.bal
    res=. (bal <@; /. mon) (<ndx;i+2) } res
  end.
end.
STOPS=: res
)
jdb_stopsetone=: 4 : 0

'name val line'=. y

msk=. ({."1 STOPS) = <name
stp=. msk # STOPS
bal=. (-.msk) # STOPS
if. SMBOTH do. val=. 0 1 end.

if. 0 = #stp do.
  if. -. x -: 0 do.
    stp=. name;0;0;(0 1 e. val) {'';line
  end.
else.
  stp=. {. stp
  for_v. val do.
    'all sel'=. (v + 1 3) { stp
    select. x
    case. 0 do.
      if. all do.
        sel=. i.NUMLINES
        all=. 0
      end.
      sel=. sel -. line
    case. 1 do.
      if. -. all do.
        sel=. ~. line, sel
      end.
    case. do.
      if. all do.
        all=. 0
        sel=. (i.NUMLINES) -. line
      else.
        if. line e. sel do.
          sel=. sel -. line
        else.
          sel=. ~. line, sel
          if. 0=#(i.NUMLINES) -. sel do.
            all=. 1
            sel=. ''
          end.
        end.
      end.
    end.
    stp=. (all;/:~sel) (v + 1 3) } stp
  end.
end.

STOPS=: stp, bal
)
jdb_stopset=: 3 : '13!:3 STOPLAST=: y'
jdb_stopsetline=: 4 : 0
x jdb_stopsetone NAME;VALENCE;y
jdb_stopwrite ''
)
jdb_stopson=: 3 : 0
'name valence codelines'=. y
val=. 2 | valence
nms=. {."1 STOPS
x=. (nms i. <,name) { STOPS, STOPNONE
'all line'=. (val + 1 3) { x
as=. (1 e. all) # codelines
ls=. /:~ ~. line
as ; ls
)
jdb_stopwrite=: 3 : 0
if. 0 e. #STOPS do. jdb_stopset '' return. end.
STOPS=: (-. (}.STOPNONE) (-:}.)"1 STOPS) # STOPS
if. 0 e. #STOPS do. jdb_stopset '' return. end.
STOPS=: /:~ ~. STOPS
nms=. {."1 STOPS
mon=. ": &.> 3 {"1 STOPS
mon=. (<'*') (I. ;1 {"1 STOPS) } mon
dyd=. ": &.> 4 {"1 STOPS
dyd=. (<'*') (I. ;2 {"1 STOPS) } dyd
jdb_stopset ; (nms,.mon,.dyd) ,&.> "1 ' :;'
)
jdb_extstops =: 3 : 0
'stopinfo autoinfo dissectinfo' =. 3 {. y
if. #stopinfo do.
  if. 1 = #$stopinfo do. stopinfo =. ,: stopinfo end.
  assert. (,5) -: }.$stopinfo

  (((<0;1 2)&{ jdb_installstops 0 3 4&{"1)/.~   1 2&{"1) stopinfo
end.

if. -. IFDISSECT do. return. end.

if. #autoinfo do.

  assert. '' -: $autoinfo
  assert. autoinfo e. 0 1
  jdebug_dissecttoggleauto_run autoinfo
end.

if. #dissectinfo do.
  if. 1 = #$dissectinfo do. dissectinfo =. ,: dissectinfo end.
  assert. (,5) -: }.$dissectinfo

  (((<0;1 2)&{ jdb_installdissectopts 0 3 4&{"1)/.~   1 2&{"1) dissectinfo
end.
)

jdb_installstops =: 4 : 0
jdebug_debugnametodisp x
x =. jdebug_splitheader jdebug_dispnametodebug x
x jdb_stoprefresh''
x jdb_installstops1 y
jdb_stopwrite''
)
jdb_installstops1 =: 4 : 0"1
'nam loc' =. x
'action val lines' =. y
assert. '' -: $val
assert. val e. 0 1
assert. '' -: $ action
assert. action e. 0 1 2
NUMLINES =: val { SMCOUNT
assert. 0 < NUMLINES [ 'selected valence is empty'
if. 0 = #lines do. lines =. i. NUMLINES end.
(action jdb_stopsetone (nam;val)&,)@<"0 lines
0 0$0
)
jdb_installdissectopts =: 4 : 0
jdebug_debugnametodisp x
x =. jdebug_splitheader jdebug_dispnametodebug x
x jdb_stoprefresh''
x jdb_installdissectopts1 y
)
jdb_installdissectopts1 =: 4 : 0"1
'nam loc' =. x
'opttbl val lines' =. y
assert. '' -: $val
assert. val e. 0 1
NUMLINES =: val { SMCOUNT
assert. 0 < NUMLINES [ 'selected valence is empty'
if. 0 = #lines do. lines =. i. NUMLINES end.
if. #opttbl do.
  if. 1 = #$opttbl do. opttbl =. ,: opttbl end.
  assert. (,2) -: }.$opttbl

  DISSECTOPTIONS =: DISSECTOPTIONS , opttbl;nam;loc;val;lines
else.


  nvmsk =. (nam;loc;val) -:"1 (1 2 3) {"1 DISSECTOPTIONS
  oldlines =. nvmsk # DISSECTOPTIONS

  oldlines =. (nlct =. -.&lines&.> 4 {"1 oldlines) (<a:;4)} oldlines

  oldlines =. (a: ~: nlct) # oldlines

  DISSECTOPTIONS =: ((-. nvmsk) # DISSECTOPTIONS) , oldlines
end.
0 0$0
)
jdb_swap=: 3 : 0
'' jdb_swap y
:
jdb_lxsoff''
old=. TABCURRENT
TABCURRENT=: new=. y

if. -. jdb_isgui'' do.
  AUTODISSECT=: {.!.0 ". 'AUTODISSECT_jdebug_'
  autodissectlocale=: 0$a:
  jdb_wd JDEBUG
  HWNDP=: jdb_wd 'qhwndp'
  p=. WINPOS>.0 0,MINWIDTH,MINHEIGHT
  if. TABCURRENT -: 'jdbnone' do.
    p=. 0 (3}) p
  else.
    jdb_wd JDEBUGP
  end.
  jdb_wd 'pmove ',(":p),';pas 0 0;pshow'

else.
  jdb_wd 'psel ',HWNDP
  if. new-:old do. return. end.
  if. #old do.
    (old,'_dun')~ 0
  end.
  if. -. (<'jdbnone') e. old;new do.
    jdb_wd 'set tabs active ', ":jdb_tabcurrent''
  else.
    hp=. HWNDP
    p=. -. new-:'jdbnone'
    fx=. 0 ". jdb_wd 'qform'
    jdb_wd JDEBUG, p#JDEBUGP
    HWNDP=: jdb_wd 'qhwndp'
    jdb_wd 'pmove ',":(3 {.fx),MINHEIGHT*p
    jdb_wd 'pas 0 0;pshow;ptop ',":PTOP
    jdb_wd 'psel ',hp,';pclose;psel ',HWNDP
  end.
  (new,'_ini')~ x
  if. -. new-:'jdbnone' do.
    jdb_wd 'set tabs active ', ":jdb_tabcurrent''
  end.
end.

if. IFDISSECT do.
  wd 'set tbar checked dissecttoggleauto ' , ": AUTODISSECT
end.

jdb_tbenable''
jdb_swapfkey''
)
TABGROUPS=: ;: 'jdbmain jdbstop jdbwatch'
DISSECTSTATUS =: (dissectchecklevel =: 4 : 0&(4 0)) 0
if. -. IFDISSECT do. _1 return. end.
if. x +. 0 > 4!:0 <'dissect_dissect_' do.

  if. fexist getscripts_j_ 'debug/dissect' do.
    load 'debug/dissect'
  end.
end.
if. 3 = 4!:0 <'dissect_dissect_' do.

  if. 0 ~: {. /: y ,: ". 'DISSECTLEVEL_dissect_' do.
    0
  else. 1
  end.
else. _1
end.
)
DTTCURR =. DTTTOGGLE =. 'These functions are defined in the debug/dissect addon, use Package Manager to get it'
DTTCURR =. 'Dissect current/cursor line' [^:(DISSECTSTATUS=1) DTTCURR
DTTTOGGLE =. 'Automatically dissect on stop' [^:(DISSECTSTATUS=1) DTTTOGGLE

DTTTBAR=: IFDISSECT # 0 : 0 rplc 'DTTCURR';DTTCURR;'DTTTOGGLE';DTTTOGGLE
set tbar add dissectcurrent "DTTCURR" "DEBUGPATH/dissect-current.png";
set tbar add dissecttoggleauto "DTTTOGGLE" "DEBUGPATH/dissect-toggle.png";
set tbar checkable dissecttoggleauto;
set tbar addsep;
)
JDEBUG=: (0 : 0 rplc 'DTTTBAR';DTTTBAR) rplc 'DEBUGPATH';(jpath '~addons/ide/qt/images')
pc jdebug escclose ptop;pn "Debug - Ctrl+H help";
cc tbar toolbar 22x22 flush;
set tbar add run "Run" "DEBUGPATH/run.png";
set tbar addsep;
set tbar add stepinto "Step into" "DEBUGPATH/stepinto.png";
set tbar add stepover "Step over" "DEBUGPATH/stepover.png";
set tbar add stepout "Step out" "DEBUGPATH/stepout.png";
set tbar addsep;
set tbar add runcursor "Run to cursor" "DEBUGPATH/runcursor.png";
set tbar add cutback "Cut back" "DEBUGPATH/cutback.png";
set tbar addsep;
set tbar add back "Back" "DEBUGPATH/goback.png";
set tbar add refresh "Refresh" "DEBUGPATH/refresh.png";
set tbar add forward "Forward" "DEBUGPATH/goforward.png";
set tbar addsep;
set tbar add stopname "Stop name at cursor" "DEBUGPATH/stopname.png";
set tbar add stopwin "Stop Manager" "DEBUGPATH/stopmanager.png";
set tbar add watchwin "Watch Manager" "DEBUGPATH/watchmanager.png";
set tbar add stack "View stack" "DEBUGPATH/stack.png";
set tbar addsep;
DTTTBAR
set tbar add clear "Clear" "DEBUGPATH/clear.png";
)
JDEBUGP=: 0 : 0
minwh 540 400;
cc tabs tab nobar;
tabnew jdbmain;
splitv;
cc lines editm readonly selectable;
set lines font fixfont;
splitsep;
splitv;
cc stack editm readonly;
set stack font fixfont;
splitsep;
cc value editm readonly;
set value font fixfont;
splitend;
splitend;
tabnew jdbstop;
bin m8h;
cc stopline button;cn "Stop Line";
cc stopall button;cn "Stop All";
bin s;
cc syslocs checkbox;cn "Show all locales";
cc stopclose button;cn "Close";
bin zhv;
cc s0 static;cn "Name:";
cc name combobox;
bin zv;
cc s1 static;cn "In:";
cc locs combobox;
bin zz;
cc slines editm readonly selectable;
set slines font fixfont;
tabnew jdbwatch;
bin m8hv;
cc s1 static; cn "Enter Watch Names:";
cc wlist editm;
set wlist font fixfont;
bin zvs1;
cc watchclear button;cn "Clear";
cc watchclose button;cn "Close";
bin s10zz;
tabend;
)
jdebug_run=: 3 : 0
jdb_swap > (y-:1) { 'jdbnone';'jdbmain'
)
jdebug_clearstops=: 3 : 0
STOPS=: 0#STOPS
jdb_lxsoff''
jdb_dbss''
jdb_lexwin''
jdb_lxson''
)
jdebug_runnext=: 3 : 0
jdb_restore''
if. MOVELINE=CURRENTLINE do.
  13!:5 ''
else.
  13!:7 MOVELINE
end.
)
jdebug_hctrl_fkey=: 3 : 0
jdb_info 'Debug Shortcuts';SHORTCUTS
)
jdebug_resize=: 3 : 0
if. #ACTIVE do.
  echo i.0 0
  jdb_setactive ACTIVE
  ACTIVE=: ''
end.
)
jdebug_tctrl_fkey=: 3 : 0
jdb_lxsoff''
PTOP=: -. PTOP
jdb_wd 'psel ',HWNDP,';ptop ',":PTOP
jdb_lxson''
)
jdebug_wctrl_fkey=: 3 : 0
jdb_lxsoff''
jdb_setactive 'term'
wd 'sm prompt *   ',jdb_dlb MOVELINE >@{ LINES
jdb_lxson''
)
jdebug_cancel=: jdebug_close=: jdb_close

jdebug_enter=: ]
jdbmain_dun=: ]
jdbmain_ini=: ]
jdbmain_stopline_button=: 3 : 0
jdb_lxsoff''
bgn=. {. 0 ". lines_select
line=. +/ LF = bgn {. lines
opt=. (line e. CODELINES) >@{ 0;2
opt jdb_stopsetline line
jdb_lexwin''
jdb_lxson''
)
jdbmain_stopname_button=: 3 : 0
jdb_lxsoff''
'ndx name'=. jdbmain_getname''
if. #name do.
  1 jdb_stoponall name
  jdb_wd 'set lines select ',":ndx
end.
jdb_lxson''
)
jdbmain_stopwin_button=: 3 : 0
jdb_lxsoff''
'ndx name'=. jdbmain_getname''
name jdb_swap 'jdbstop'
jdb_lxson''
)
jdebug_lines_button=: jdebug_stepover_button
jdbmain_getname=: 3 : 0
sel=. 0 ". lines_select
end=. I. LF=lines,LF
bgn=. 0, 1+}:end
if. sel e. bgn ,. end do.
  'ndx name'=. sel jdb_getnamesat lines
else.
  'ndx name'=. sel jdb_getnameat lines
end.
)
jdb_writelines=: 4 : 0
len=. 0, +/\ 1 + # &> y
jdb_wd 'set lines text *',jdb_listboxed y
jdb_wd 'setscroll lines ',":SCROLL
jdb_wd 'setfocus lines'
if. x do.
  jdb_wd 'set lines select ',": 0 _1 + (MOVELINE+0 1){len
end.
jdb_minsize''
)
jdbnone_dun=: ]
jdbnone_ini=: ]
jdbnone_stopwin_button=: 3 : 0
'' jdb_swap 'jdbstop'
jdb_lxson''
)

jdbnone_stopname_button=: empty
CX=: <'Current execution'

DEBUGNAMESUFFIX =: 'd4B7g0'
DISPNAMESUFFIX =: ' (header)'
jdbstop_dun=: ]
jdbstop_ini=: 3 : 0

if. jdb_inactive'' do.
  SMLOC=: <'base'
  0 jdebug_syslocs_button ''
  STNAMES=: i.0 2
  jdb_stoplocaleset SMLOC
else.
  SMLOC=: CX
  0 jdebug_syslocs_button ~. LOCALE, <'base'
  nms=. {."1 STACK
  nms=. ~. (jdb_boxxopen y), nms
  r=. ([:{.(_2&{.@I.@('_'=])))&.>nms
  p=. ;('_'={:)&.>nms
  nms1=. (r+&.>(-.p)*&.>#&.>nms){.&.>nms
  SMNAMES=: nms1 ,. LOCALE(I. -.;p)} ([:}.(}:))&.>r}.&.> nms
  STNAMES=: SMNAMES
  0 jdb_stopswritedef SMNAMES
end.
)
jdbstop_stopall_button=: 3 : 0
jdb_lxsoff''
ndx=. {. _1,~ 0 ". name_select
if. _1=ndx do. jdb_lxson'' return. end.
nam=. {. ndx { SMNAMES
2 jdb_stoponall nam
jdb_stoprefresh {: 0 ". slines_select
jdb_lxson''
)
jdbstop_stopline_button=: 3 : 0
jdb_lxsoff''
if. -. LF e. slines do.
  jdbstop_stopall_button''
else.
  'bgn end'=. 2{._1 _1,~ 0 ". slines_select
  num=. +/ LF = bgn {. slines
  val=. num > {. SMCOUNT
  line=. num - val * 1 + {.SMCOUNT
  nam=. {. SMNDX { SMNAMES
  NUMLINES=: val { SMCOUNT
  2 jdb_stopsetone nam,val;line
  jdb_stopwrite''
  jdb_stoprefresh end
end.
jdb_lxson''
)
jdbstop_stopname_button=: 3 : 0
jdb_lxsoff''
if. 0 e. $j=. jdbstop_getnameat'' do. jdb_lxson'' return. end.
'pos name fullid'=. j

if. fullid -: jdbstop_getcurrentname'' do.
  jdbstop_stopall_button''
else.
  1 jdb_stoponall name
  jdb_wd 'set slines select ',":pos
  if. SMLOC-:CX do.
    SMNAMES=: ~. SMNAMES, fullid
    jdb_wd 'set name items ',jdb_listboxed {."1 jdebug_debugnametodisp SMNAMES
  end.
end.
jdb_lxson''
)
jdbstop_stopwin_button=: 3 : 0
jdb_lxsoff''
if. 0 e. $j=. jdbstop_getnameat'' do. jdb_lxson'' return. end.

'pos name fullid'=. j
if. fullid -: SMNDX { SMNAMES do. jdb_lxson'' return. end.

'rep both count'=. jdb_stoprep fullid
if. 0=#rep do.
  jdb_info 'No definition for name:',LF,LF,name
  jdb_lxson'' return.
end.

if. fullid e. SMNAMES do.
  nms=. SMNAMES
  ndx=. SMNAMES i. fullid
else.
  nms=. ~. fullid, SMNAMES
  ndx=. 0
end.

jdb_stopswritedefone rep;both;count;ndx;<nms
jdb_lxson''
)
jdebug_name_select=: 3 : 0
jdb_lxsoff''
ndx=. {. _1,~ 0 ". name_select
if. _1=ndx do. jdb_lxson'' return. end.
jdb_wd 'set locs select ',": SMLOCS i. SMLOC
if. ndx = #SMNAMES do.





  'dbgnames dispnames' =. jdebug_debugnametodisp SMNAMES
  jdb_wd 'set name items ',jdb_listboxed {."1 dispnames
  jdb_wd 'set name select ', ": SMNDX
  jdb_lxson''
  return.
end.
if. ndx ~: SMNDX do.



  if. (ndx { SMNAMES) -.@-: fullname =. jdebug_splitheader ndx { SMNAMES do.



    SMNAMES =: (2 ndx} (#SMNAMES)$1) # SMNAMES
    'dbgnames dispnames' =. jdebug_debugnametodisp fullname (>:ndx)} SMNAMES
    SMNAMES =: dbgnames
    ndx =. SMNAMES i. fullname
    jdb_wd 'set name items ',jdb_listboxed {."1 dispnames
    jdb_wd 'set name select ', ": ndx
  end.
  'rep both count'=. jdb_stoprep fullname
  if. 0=#rep do.
    j=. 'Unable to get representation of:', LF, LF
    jdb_info j, 0 >@{ ndx { SMNAMES
    'dbgnames dispnames' =. jdebug_debugnametodisp (<<<ndx) { SMNAMES
    SMNAMES =: dbgnames
    SMNDX=: 0
    jdb_wd 'set name items ',jdb_listboxed {."1 dispnames
    jdb_wd 'set name select ', ": SMNDX
  else.
    if. DISPNAMESUFFIX ([ -: -@#@[ {. ]) name do.


      rep =. DEBUGNAMESUFFIX&(taketo , takeafter)&.> rep
    end.
    jdb_wd 'set slines text *',jdb_listboxed rep
    SMNDX=: ndx
    NMC=: 4!:0 fullname
    SMBOTH=: both
    SMCOUNT=: count
  end.
end.
jdb_lxson''
)
jdebug_locs_select=: 3 : 0
jdb_lxsoff''
if. SMLOCS -.@e.~ <locs do.


  jdb_wd 'set locs items ',jdb_toDEL SMLOCS
  jdb_wd 'set locs select ' , ": SMLOCS i. SMLOC
else.
  jdb_stoplocaleset locs
end.
jdb_lxson''
)

SYSTEMLOCALES =: ;: 'z'
SYSTEMPREFIXES =: ;: 'dissect lint j'
jdebug_syslocs_button=: 3 : 0
(0 ". syslocs) jdebug_syslocs_button y
:
lc=. (18!:1[0), 18!:1[1
if. 0 = x do.
  lc =. (#~   [: -. [: +./ SYSTEMPREFIXES ([ -: #@[ {. ])&>/ ]) lc -. SYSTEMLOCALES
  lc =. (#~   *@#@(-.&'0123456789')@>) lc
end.
jdb_wd 'set locs items ',jdb_toDEL SMLOCS =: (,   lc -. ]) SMLOC , y
jdb_wd 'set locs select 0'
0 0$0
)
jdbstop_getcurrentname=: 3 : 0
ndx=. 0 ". name_select
ndx { SMNAMES
)
jdbstop_getnameat=: 3 : 0

'pos name'=. (0 ". slines_select) jdb_getnameat slines

if. 0 = #name do. '' return. end.
loc=. {: SMNDX { SMNAMES
fullid=. loc jdb_fullname name
pos;name;<fullid
)
jdb_stoplocaleset=: 3 : 0
bloc=. jdb_boxopen y
ndx=. SMLOCS i. bloc
idx=. 0

if. bloc -: CX do.
  idx jdb_stopswritedef STNAMES
else.
  18!:4 bloc
  ids=. 4!:1 [ 1 2 3
  18!:4 <'jdebug'
  idx jdb_stopswritedef ids ,. bloc
end.

jdb_wd 'set locs select ',": ndx
SMLOC=: bloc
)
jdb_stopname=: 3 : 0
if. 0 = #y do. 0 return. end.
jdb_wd 'set locs items;setenable locs 0'
y=. jdb_fullname y
nms=. ~. y,SMNAMES,NAME;LOCALE
if. 0 jdb_stopswritedef nms do.
  STNAMES=: ~. y, SMNAMES
  1
else.
  0
end.
)
jdb_stoprefresh=: 3 : 0
(jdbstop_getcurrentname'') jdb_stoprefresh y
:
'rep both count'=. jdb_stoprep x
if. #y do.
  srep=. jdb_listboxed rep
  jdb_wd 'set slines text *', srep
  sel=. 2 $ y
  jdb_wd 'set slines select ',":sel, 0
  jdb_wd 'setfocus slines'
end.
SMBOTH=: both
SMCOUNT=: count
)
jdb_stoprep=: 3 : 0

name=. jdb_boxopen y
rep=. jdb_getdrep name
lname=. ; name ,&.> '_'
both=. 0

if. 0=#rep do. '';0;0 0 return. end.

tac=. -. jdb_isexplicit lname
cls=. 4!:0 <lname
'cls rep0 rep1'=. (tac,cls) jdb_boxrep rep

if. rep0 -: rep1 do.
  if. cls=4 do.
    rep0=. ''
  else.
    both=. 1
    rep1=. ''
  end.
end.

cod0=. jdb_codelines rep0
cod1=. jdb_codelines rep1
num0=. #rep0
num1=. #rep1

if. num0 do.
  stp0=. jdb_stopgetone name,0;num0;cod0
  r=. stp0 ,&.> jdb_indexit rep0
else.
  r=. ''
end.

if. num1 do.
  stp1=. jdb_stopgetone name,1;num1;cod1
  r=. r, <' [:] ',40#'-'
  r=. r, stp1 ,&.> jdb_indexit rep1
end.

r; both ; num0, num1
)
jdb_stopswritedef=: 4 : 0
if. 0 e. #y do.
  jdb_stopswritedefone ''
else.
  'rep both count'=. jdb_stoprep x { y
  jdb_stopswritedefone rep;both;count;x;<y
end.
)
jdb_stopswritedefone=: 3 : 0
if. 0 e. #y do.
  jdb_wd 'set name items;set slines text'
  SMNAMES=: i.0 2
  SMNDX=: 0
  SMCOUNT=: 0 0
  0
else.
  'rep both count ndx nms'=. y
  slines_select=: ''
  'dbgnames dispnames' =. jdebug_debugnametodisp nms
  name_select=: ": SMNDX =: dbgnames i. ndx { nms
  SMNAMES =: dbgnames
  jdb_wd 'set name items ',jdb_listboxed {."1 dispnames
  jdb_wd 'set name select ', ": SMNDX
  jdb_wd 'set name select ',name_select
  jdb_wd 'set slines text *', jdb_listboxed rep
  jdb_wd 'setenable name 1;setenable slines 1'
  SMBOTH=: both
  SMCOUNT=: count
  *#rep
end.
)
jdebug_debugnametodisp =: ,&.>/ @: ((3 : 0)/.~   1&{"1) @: (,:^:(1=#@$))
dispnms =. nms =. 0 {"1 y
loc =. (<0 1) { y
cocurrent loc
ids=. 4!:1 [ 1 2 3
cocurrent <'jdebug'
if. #dids =. (#~   DEBUGNAMESUFFIX&([ -:"1 -@#@[ {.&> ])) ids do.


  oids =. (-#DEBUGNAMESUFFIX) }.&.> dids


  if. #deadoids =. (#~  loc&jdebug_ismultiline) oids do.
    deaddids =. ,&DEBUGNAMESUFFIX&.> deadoids
    dids =. dids -. deaddids
    oids =. oids -. deadoids
    nms =. nms -. deaddids
    4!:55 ,&(,&'_')&.>/"1 deaddids ,. loc
  end.

  hids =. ,&' (header)'&.> oids

  dispnms =. (hids,oids,nms) {~ (oids,dids,nms) i. nms
end.
(nms,.loc) ,&< (dispnms,.loc)
)

jdebug_dispnametodebug =: 3 : 0
nm =. 0 {:: y
if. DISPNAMESUFFIX ([ -: -@#@[ {. ]) nm do. nm =. (-#DISPNAMESUFFIX) }. nm
elseif.
cocurrent (1 { y)
ids =. ({.nm) 4!:1 [ 1 2 3
cocurrent <'jdebug'
(<nm,DEBUGNAMESUFFIX) e. ids
do.
  nm =. nm,DEBUGNAMESUFFIX
end.
nm ; 1 { y
)
jdebug_splitheader =: 3 : 0
nm =. 0 {:: y
loc =. 1 { y
ld =. 5!:5 <nm,'__loc'
if. (LF,')') -: _2 {. ld do.
  'l1 ln' =. LF (taketo ; takeafter) ld
  if. 3 < # l1w =. ;: l1 do.
    if. 1 = #cox =. I. 2 (;:':0')&-:\ l1w do.
      cox =. {. cox
      if. 4 > exptype =. ('1234' ;&,"0 '0') i. ((_1 1 + cox) { l1w,a:)  do.
        keepcom =. 9!:40''
        9!:41 'NB.' +./@:E. ln

        ((dnm =. nm,DEBUGNAMESUFFIX),'__loc') =: (>:exptype) : (}: ln)

        try.
          ". (nm,'__loc =: ') , ;:^:_1 ('';dnm;'') (_1 0 1 + cox)} l1w

          nm =. dnm
        catch.

          4!:55 <dnm,'__loc'
        end.

        9!:41 keepcom
      end.
    end.
  end.
elseif. (dnm =. nm,DEBUGNAMESUFFIX) +./@:E. ld do. nm =. dnm
end.
nm ; loc
)
jdebug_ismultiline =: 4 : 0"0
(LF,')') -: _2 {. 5!:5 <(>y),'__x'
)
jdebug_locs_button=: jdebug_locs_select
jdebug_name_button=: jdebug_name_select
jdebug_stopall_button=: jdbstop_stopall_button
jdebug_stopclose_button=: jdebug_mainwin
j=. 0 : 0
run       1 0 0 0
stepinto  1 0 0 0
stepover  1 0 0 0
stepout   1 0 0 0
runcursor 1 0 0 0
cutback   1 0 0 0
back      1 0 0 0
refresh   1 0 0 0
forward   1 0 0 0
stopname  1 1 0 0
stopwin   1 1 1 1
watchwin  1 1 0 1
stack     1 1 1 1
dissectcurrent 1 1 0 0
dissecttoggleauto 1 1 1 1
clear     1 1 1 0
)

f=. (1 + i.&' ') ({.;".@}.) ]
j=. f ;._2 j
j=. j #~ IFDISSECT >: ('dissect' -: 7 {.]) &> {."1 j
nms=. ';set tbar enable '&, &.> {."1 j
nms1=. jdb_deb &.> {."1 j
vls=. >{:"1 j

a=. }.;nms (,":) &.> 0 {"1 vls
b=. }.;nms (,":) &.> 1 {"1 vls
c=. }.;nms (,":) &.> 2 {"1 vls
d=. }.;nms (,":) &.> 3 {"1 vls
TBENABLE=: a;b;c;d

a=. <nms1 ,. <("0) 0 {"1 vls
b=. <nms1 ,. <("0) 1 {"1 vls
c=. <nms1 ,. <("0) 2 {"1 vls
d=. <nms1 ,. <("0) 3 {"1 vls
TBENABLE1=: a,b,c,d
jdb_tabcurrent=: 3 : 'TABGROUPS i. <TABCURRENT'
jdb_tbenable=: 3 : 0
jdb_wd (jdb_tabcurrent'') >@{ TBENABLE
)
jdebug_back_button=: 3 : 0
MOVELINE=: jdb_nextline _1
jdb_lexwin''
)
jdebug_clear_button=: 3 : 0
jdb_clear''
)
jdebug_cutback_run=: 3 : 0
jdb_restore''
13!:19 ''
)
jdebug_forward_button=: 3 : 0
MOVELINE=: jdb_nextline 1
jdb_lexwin''
)
jdebug_help_button=: 3 : 0
jdb_lxsoff''
jdb_stopread''
jdb_dbss''
if. 0 -: htmlhelp_j_ :: 0: 'user/debugs.htm' do.
  jdb_info 'Unable to access debug help'
end.
jdb_stopwrite''
jdb_lxson''
)
jdebug_stack_button=: 3 : 0
stk=. 2}.13!:13''
if. 0 e. $stk do.
  jdb_info 'Nothing on the stack' return.
end.
hdr=. ;:'name en ln nc args locals susp'
stk=. 1 1 1 1 0 0 1 1 1 #"1 stk
stk=. hdr, ": &.> stk
wds=. ({:@$@":@,.)"1 |: stk
len=. 40 >.<.-: 120 - +/8, 4 {. wds
tc=. (len+1)&<.@$ {.!.'.' ({.~ len&<.@$)
r=. tc@": &.> stk
r=. ,(":r),.LF
s=. {:a.
jdb_wd 'textview *',s,'stack',s,s,r
i. 0 0
)
jdebug_refresh_button=: 3 : 0
MOVELINE=: CURRENTLINE
jdb_lexwin''
)
jdebug_run_run=: 3 : 0
jdb_restore''
if. MOVELINE=CURRENTLINE do.
  13!:4 ''
else.
  13!:7 MOVELINE
end.
)
jdebug_runcursor_run=: 3 : 0
line=. jdb_getcursorline''
if. line <: CURRENTLINE do.
  jdb_info 'Line selected should be after current line' return.
elseif. line > >./ CODELINES do.
  jdb_info 'Cannot stop on selected line' return.
end.
jdb_restore''
names=. {."1 STACK
CUTNAMES=: LOCALE jdb_addlocale &.> names
CUTLINES=: ; 2 {"1 STACK
0 jdb_stopsetline CURRENTLINE + i. line - CURRENTLINE
1 jdb_stopsetline line
jdb_restore''
13!:4''
)
jdebug_dissectcurrent_run=: 3 : 0
if. DISSECTSTATUS ~: 1 do. DISSECTSTATUS =: dissectchecklevel 1 end.
select. DISSECTSTATUS
case. 0 do.
  wdinfo 'Out-of-date addon';'The addon, debug/dissect, is out of date.  Use Tools|Package Manager to update it.'
  empty''
case. _1 do.
  wdinfo 'Dissect Current/Cursor Line';'This feature requires the debug/dissect addon.  Use Tools|Package Manager to install the addon.'
  empty''
case. do.
  if. 0 = #line =. y do.
    if. 0 = #line=. jdb_getcursorline'' do.
      line =, CURRENTLINE
    end.
  end.

  dissectopts =. 12 14 {~ *#y
  if. #dos =. (((NAME;LOCALE,<VALENCE) -:"1 (1 2 3)&{"1) # 0 4&{"1) DISSECTOPTIONS do.
    if. #dos =. (#~    line e.&> {:"1) dos do.

      dopts =. ; {."1 dos
      if. +./ amsk =. (<'auto') = {."1 dopts do.

        if. (*#y) *. (+./ ('auto';'0') -:&:(,&.>)"1 amsk # dopts) do.

          dissectopts =. dopts =. ''
        else.
          dopts =. (-. amsk) # dopts
        end.
      end.
      if. #dopts do. dissectopts =. dissectopts ;< dopts end.
    end.
  end.

  jdb_restore (*#dissectopts) # (line { LINES) , <dissectopts
end.
)
jdebug_dissecttoggleauto_run=: 3 : 0
if. DISSECTSTATUS ~: 1 do. DISSECTSTATUS =: dissectchecklevel 1 end.
select. DISSECTSTATUS
case. 0 do.
  wdinfo 'Out-of-date addon';'The addon, debug/dissect, is out of date.  Use Tools|Package Manager to update it.'
case. _1 do.
  wdinfo 'Toggle Autodissect';'This feature requires the debug/dissect addon.  Use Tools|Package Manager to install the addon.'
case. do.
  if. #y do.
    wd 'psel jdebug'
    AUTODISSECT =: y
  else.
    AUTODISSECT =: -. AUTODISSECT
  end.
end.
wd 'set tbar checked dissecttoggleauto ' , ": AUTODISSECT
empty''
)
jdebug_stepout_run=: 3 : 0
jdb_restore''
if. MOVELINE=CURRENTLINE do.
  13!:22 ''
else.
  MOVELINE 13!:22 ''
end.
)
jdebug_stepover_run=: 3 : 0
jdb_restore''
if. MOVELINE=CURRENTLINE do.
  13!:20 ''
else.
  MOVELINE 13!:20 ''
end.
)
jdebug_stepinto_run=: 3 : 0
jdb_restore''
if. MOVELINE=CURRENTLINE do.
  13!:21 ''
else.
  MOVELINE 13!:21 ''
end.
)
jdebug_main=: 3 : 0
if. jdb_inactive'' do.
  jdb_swap 'jdbnone'
else.
  jdb_swap 'jdbmain'
  if. #jdb_getstack'' do.
    jdb_lexwin''
  end.
end.
jdb_lxson''
)
jdebug_watchwin_button=: 3 : 0
jdb_lxsoff''
jdb_swap 'jdbwatch'
jdb_lxson''
)
jdebug_mainwin=: 3 : 0
if. jdb_inactive'' do.
  jdb_swap 'jdbnone'
else.
  jdb_swap 'jdbmain'
  if. #jdb_getstack'' do.
    jdb_lexwin''
  end.
end.
jdb_lxson''
)
jdebug_stopline_button=: 3 : '(TABCURRENT,''_stopline_button'')~0'
jdebug_stopname_button=: 3 : '(TABCURRENT,''_stopname_button'')~0'
jdebug_stopwin_button=: 3 : '(TABCURRENT,''_stopwin_button'')~0'
jdebug_stepover_button=: jdebug_stepover_run
jdebug_stepinto_button=: jdebug_stepinto_run
jdebug_stepout_button=: jdebug_stepout_run
jdebug_cutback_button=: jdebug_cutback_run
jdebug_runcursor_button=: jdebug_runcursor_run
jdebug_run_button=: jdebug_run_run
jdebug_dissectcurrent_button=: jdebug_dissectcurrent_run
jdebug_dissecttoggleauto_button=: jdebug_dissecttoggleauto_run
jdebug_jctrl_fkey=: 3 : 'lab_jlab_ 0'
jdbwatch_dun=: 3 : 0
if. 0 ~: 4!:0 <'wlist' do. return. end.

txt=. ' ' (I. wlist=LF)} wlist
nms=. jdb_cutopen txt
nmc=. (4!:0 :: _2:)"0 nms
if. _2 e. nmc do.
  bad=. nmc = _2
  t=. ;: ^:_1 bad # nms
  jdb_info 'Invalid watch name',((1<+/bad)#'s'),': ',t
  return.
end.
WATCH=: nms
)
jdbwatch_ini=: 3 : 0
txt=. jdb_tolist jdb_cutopen WATCH
jdb_wd 'set wlist text *',txt
jdb_wd 'setfocus wlist'
)
jdebug_watchclear_button=: 3 : 0
jdb_wd 'set wlist text'
jdb_lxson''
)
jdbwatch_stopwin_button=: 3 : 0
'' jdb_swap 'jdbstop'
jdb_lxson''
)
jdebug_watchclose_button=: jdebug_mainwin
