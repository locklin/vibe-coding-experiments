coclass 'qtopengl'

NB. gl2 overlay on opengl
NB. all drawing commands should be used inside paintz event only

chkgl2=: 13!:8@3:^:(0&<)@>@{.

NB. =========================================================
gl_arc=: chkgl2 @: (('"',libjqt,'" gl_arc >',(IFWIN#'+'),' i *i') cd <) "1
gl_brush=: chkgl2 @: (('"',libjqt,'" gl_brush >',(IFWIN#'+'),' i')&cd bind '') "1
gl_brushnull=: chkgl2 @: (('"',libjqt,'" gl_brushnull >',(IFWIN#'+'),' i')&cd bind '') "1
gl_capture=: chkgl2 @: (('"',libjqt,'" gl_capture >',(IFWIN#'+'),' i i')&cd) "1
gl_caret=: chkgl2 @: (('"',libjqt,'" gl_caret >',(IFWIN#'+'),' i *i') cd <) "1
gl_clear=: chkgl2 @: (('"',libjqt,'" gl_clear >',(IFWIN#'+'),' i')&cd bind '') "1
gl_clip=: chkgl2 @: (('"',libjqt,'" gl_clip >',(IFWIN#'+'),' i *i') cd <) "1
gl_clipreset=: chkgl2 @: (('"',libjqt,'" gl_clipreset >',(IFWIN#'+'),' i')&cd bind '') "1
gl_cmds=: chkgl2 @: (('"',libjqt,'" gl_cmds >',(IFWIN#'+'),' i *i i') cd (;#)) "1
gl_cursor=: chkgl2 @: (('"',libjqt,'" gl_cursor >',(IFWIN#'+'),' i i')&cd) "1
gl_ellipse=: chkgl2 @: (('"',libjqt,'" gl_ellipse >',(IFWIN#'+'),' i *i') cd <@:<.) "1
gl_fill=: chkgl2 @: (('"',libjqt,'" gl_fill >',(IFWIN#'+'),' i *i') cd <@:<.) "1
gl_font=: chkgl2 @: (('"',libjqt,'" gl_font >',(IFWIN#'+'),' i *c') cd <@,) "1
gl_font2=: chkgl2 @: (('"',libjqt,'" gl_font2 >',(IFWIN#'+'),' i *i i') cd (;#)@:<.) "1
gl_fontangle=: chkgl2 @: (('"',libjqt,'" gl_fontangle >',(IFWIN#'+'),' i i')&cd) "1
gl_lines=: chkgl2 @: (('"',libjqt,'" gl_lines >',(IFWIN#'+'),' i *i i') cd (;#)) "1
gl_nodblbuf=: chkgl2 @: (('"',libjqt,'" gl_nodblbuf >',(IFWIN#'+'),' i i') cd {.@(,&0)) "1
gl_paint=: chkgl2 @: (('"',libjqt,'" gl_paint >',(IFWIN#'+'),' i')&cd bind '') "1
gl_paintx=: chkgl2 @: (('"',libjqt,'" gl_paintx >',(IFWIN#'+'),' i')&cd bind '') "1
gl_pen=: chkgl2 @: (('"',libjqt,'" gl_pen >',(IFWIN#'+'),' i *i') cd <@:(2&{.)) "1
gl_pie=: chkgl2 @: (('"',libjqt,'" gl_pie >',(IFWIN#'+'),' i *i') cd <) "1
gl_pixel=: chkgl2 @: (('"',libjqt,'" gl_pixel >',(IFWIN#'+'),' i *i') cd <) "1
gl_pixels=: chkgl2 @: (('"',libjqt,'" gl_pixels >',(IFWIN#'+'),' i *i i') cd (;#)@:<.) "1
gl_pixelsx=: chkgl2 @: (('"',libjqt,'" gl_pixelsx >',(IFWIN#'+'),' i *i') cd <@:<.) "1
gl_polygon=: chkgl2 @: (('"',libjqt,'" gl_polygon >',(IFWIN#'+'),' i *i i') cd (;#)@:<.) "1
gl_rect=: chkgl2 @: (('"',libjqt,'" gl_rect >',(IFWIN#'+'),' i *i') cd <) "1
gl_rgb=: chkgl2 @: (('"',libjqt,'" gl_rgb >',(IFWIN#'+'),' i *i') cd <@:<.) "1
gl_rgba=: chkgl2 @: (('"',libjqt,'" gl_rgba >',(IFWIN#'+'),' i *i') cd <@:<.) "1
gl_sel=: chkgl2 @: (('"',libjqt,'" gl_sel >',(IFWIN#'+'),' i x')&cd) "1
gl_sel2=: chkgl2 @: (('"',libjqt,'" gl_sel2 >',(IFWIN#'+'),' i *c') cd <@,) "1
gl_text=: chkgl2 @: (('"',libjqt,'" gl_text >',(IFWIN#'+'),' i *c') cd <@,) "1
gl_textcolor=: chkgl2 @: (('"',libjqt,'" gl_textcolor >',(IFWIN#'+'),' i')&cd bind '') "1
gl_textxy=: chkgl2 @: (('"',libjqt,'" gl_textxy >',(IFWIN#'+'),' i *i') cd <@:<.) "1
gl_waitgl=: chkgl2 @: (('"',libjqt,'" gl_waitgl >',(IFWIN#'+'),' i')&cd bind '') "1
gl_waitnative=: chkgl2 @: (('"',libjqt,'" gl_waitnative >',(IFWIN#'+'),' i')&cd bind '') "1
gl_windoworg=: chkgl2 @: (('"',libjqt,'" gl_windoworg >',(IFWIN#'+'),' i *i') cd <@:<.) "1
gl_updategl=: chkgl2 @: (('"',libjqt,'" gl_updategl >',(IFWIN#'+'),' i x')&cd) "1

gl_setlocale=: chkgl2 @: (('"',libjqt,'" gl_setlocale >',(IFWIN#'+'),' i *c') cd <@,@>) "1

NB. =========================================================
gl_qhandles=: 3 : 0"1
hs=. 3#2-2
chkgl2 cdrc=. ('"',libjqt,'" gl_qhandles  ',(IFWIN#'+'),' i *x') cd <hs
1{::cdrc
)

NB. =========================================================
gl_qwh=: 3 : 0"1
wh=. 2#2-2
chkgl2 cdrc=. ('"',libjqt,'" gl_qwh  ',(IFWIN#'+'),' i *i') cd <wh
1{::cdrc
)

NB. =========================================================
NB. return matrix of pixels
NB. wh is limited to pixmap size
NB. -1 in w or h means read to end
gl_qpixelm=: 3 : 0"1
n=. */ 2{.2}.y
pix=. n#2-2
shape=. 2#2-2
chkgl2 cdrc=. ('"',libjqt,'" gl_qpixelm  ',(IFWIN#'+'),' i *i *i *i') cd y;shape;pix
(2&{:: $ 3&{::) cdrc
)

NB. =========================================================
NB. result is in opengl_ form
NB. pixels top to bottom, RGB24
NB. TODO
gl_qpixels=: 3 : 0"1
n=. */ 2{.2}.y
pix=. n#2-2
chkgl2 cdrc=. ('"',libjqt,'" gl_qpixels  ',(IFWIN#'+'),' i *i *i') cd y;pix
2{::cdrc
)

NB. =========================================================
NB. TODO
gl_qextent=: 3 : 0"1
wh=. 2#2-2
chkgl2 cdrc=. ('"',libjqt,'" gl_qextent  ',(IFWIN#'+'),' i *c *i') cd (,y);wh
2{::cdrc
)

NB. =========================================================
NB. TODO
gl_qextentw=: 3 : 0"1
y=. y,(LF~:{:y)#LF [ y=. ,y
w=. (+/LF=y)#2-2
chkgl2 cdrc=. ('"',libjqt,'" gl_qextentw  ',(IFWIN#'+'),' i *c *i') cd y;w
2{::cdrc
)

NB. =========================================================
NB. font information: Height, Ascent, Descent, InternalLeading, ExternalLeading, AverageCharWidth, MaxCharWidth
NB. TODO
gl_qtextmetrics=: 3 : 0"1
tm=. 7#2-2
chkgl2 cdrc=. ('"',libjqt,'" gl_qtextmetrics  ',(IFWIN#'+'),' i *i') cd tm
1{::cdrc
)
