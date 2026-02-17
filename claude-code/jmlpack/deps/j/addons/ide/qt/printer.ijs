coclass 'qtprinter'

NB. gl2 on printer
NB. all drawing commands should be used for printer only

chkgl2=: 13!:8@3:^:(0&<)@>@{.

NB. =========================================================
glzarc=: chkgl2 @: (('"',libjqt,'" glzarc >',(IFWIN#'+'),' i *i') cd <) "1
glzbrush=: chkgl2 @: (('"',libjqt,'" glzbrush >',(IFWIN#'+'),' i')&cd bind '') "1
glzbrushnull=: chkgl2 @: (('"',libjqt,'" glzbrushnull >',(IFWIN#'+'),' i')&cd bind '') "1
glzclear=: chkgl2 @: (('"',libjqt,'" glzclear >',(IFWIN#'+'),' i')&cd bind '') "1
glzclip=: chkgl2 @: (('"',libjqt,'" glzclip >',(IFWIN#'+'),' i *i') cd <) "1
glzclipreset=: chkgl2 @: (('"',libjqt,'" glzclipreset >',(IFWIN#'+'),' i')&cd bind '') "1
glzcmds=: chkgl2 @: (('"',libjqt,'" glzcmds >',(IFWIN#'+'),' i *i i') cd (;#)) "1
glzellipse=: chkgl2 @: (('"',libjqt,'" glzellipse >',(IFWIN#'+'),' i *i') cd <@:<.) "1
glzfill=: chkgl2 @: (('"',libjqt,'" glzfill >',(IFWIN#'+'),' i *i') cd <@:<.) "1
glzfont=: chkgl2 @: (('"',libjqt,'" glzfont >',(IFWIN#'+'),' i *c') cd <@,) "1
glzfont2=: chkgl2 @: (('"',libjqt,'" glzfont2 >',(IFWIN#'+'),' i *i i') cd (;#)@:<.) "1
glzfontangle=: chkgl2 @: (('"',libjqt,'" glzfontangle >',(IFWIN#'+'),' i i')&cd) "1
glzfontextent=: chkgl2 @: (('"',libjqt,'" glzfontextent >',(IFWIN#'+'),' i *c') cd <@,) "1
glzlines=: chkgl2 @: (('"',libjqt,'" glzlines >',(IFWIN#'+'),' i *i i') cd (;#)) "1
glznodblbuf=: chkgl2 @: (('"',libjqt,'" glznodblbuf >',(IFWIN#'+'),' i i') cd {.@(,&0)) "1
glzpen=: chkgl2 @: (('"',libjqt,'" glzpen >',(IFWIN#'+'),' i *i') cd <@:(2&{.)) "1
glzpie=: chkgl2 @: (('"',libjqt,'" glzpie >',(IFWIN#'+'),' i *i') cd <) "1
glzpixel=: chkgl2 @: (('"',libjqt,'" glzpixel >',(IFWIN#'+'),' i *i') cd <) "1
glzpixels=: chkgl2 @: (('"',libjqt,'" glzpixels >',(IFWIN#'+'),' i *i i') cd (;#)@:<.) "1
glzpixelsx=: chkgl2 @: (('"',libjqt,'" glzpixelsx >',(IFWIN#'+'),' i *i') cd <@:<.) "1
glzpolygon=: chkgl2 @: (('"',libjqt,'" glzpolygon >',(IFWIN#'+'),' i *i i') cd (;#)@:<.) "1
glzrect=: chkgl2 @: (('"',libjqt,'" glzrect >',(IFWIN#'+'),' i *i') cd <) "1
glzrgb=: chkgl2 @: (('"',libjqt,'" glzrgb >',(IFWIN#'+'),' i *i') cd <@:<.) "1
glzrgba=: chkgl2 @: (('"',libjqt,'" glzrgba >',(IFWIN#'+'),' i *i') cd <@:<.) "1
glztext=: chkgl2 @: (('"',libjqt,'" glztext >',(IFWIN#'+'),' i *c') cd <@,) "1
glztextcolor=: chkgl2 @: (('"',libjqt,'" glztextcolor >',(IFWIN#'+'),' i')&cd bind '') "1
glztextxy=: chkgl2 @: (('"',libjqt,'" glztextxy >',(IFWIN#'+'),' i *i') cd <@:<.) "1
glzwindoworg=: chkgl2 @: (('"',libjqt,'" glzwindoworg >',(IFWIN#'+'),' i *i') cd <@:<.) "1

glzresolution=: chkgl2 @: (('"',libjqt,'" glzresolution >',(IFWIN#'+'),' i i')&cd) "1
glzcolormode=: chkgl2 @: (('"',libjqt,'" glzcolormode >',(IFWIN#'+'),' i i')&cd) "1
glzduplexmode=: chkgl2 @: (('"',libjqt,'" glzduplexmode >',(IFWIN#'+'),' i i')&cd) "1
glzorientation=: chkgl2 @: (('"',libjqt,'" glzorientation >',(IFWIN#'+'),' i i')&cd) "1
glzoutputformat=: chkgl2 @: (('"',libjqt,'" glzoutputformat >',(IFWIN#'+'),' i i')&cd) "1
glzpageorder=: chkgl2 @: (('"',libjqt,'" glzpageorder >',(IFWIN#'+'),' i i')&cd) "1
glzpapersize=: chkgl2 @: (('"',libjqt,'" glzpapersize >',(IFWIN#'+'),' i i')&cd) "1
glzpapersource=: chkgl2 @: (('"',libjqt,'" glzpapersource >',(IFWIN#'+'),' i i')&cd) "1

glzscale=: chkgl2 @: (('"',libjqt,'" glzscale >',(IFWIN#'+'),' i *f') cd <) "1

glzabortdoc=: chkgl2 @: (('"',libjqt,'" glzabortdoc >',(IFWIN#'+'),' i')&cd bind '') "1
glzenddoc=: chkgl2 @: (('"',libjqt,'" glzenddoc >',(IFWIN#'+'),' i')&cd bind '') "1
glznewpage=: chkgl2 @: (('"',libjqt,'" glznewpage >',(IFWIN#'+'),' i')&cd bind '') "1
glzprinter=: chkgl2 @: (('"',libjqt,'" glzprinter >',(IFWIN#'+'),' i *c') cd <@,) "1
glzstartdoc=: chkgl2 @: (('"',libjqt,'" glzstartdoc >',(IFWIN#'+'),' i *c *c') cd 2: {. boxopen) "1

glzinitprinter=: chkgl2 @: (('"',libjqt,'" glzinitprinter >',(IFWIN#'+'),' i')&cd bind '') "1

glzqresolution=: (('"',libjqt,'" glzqresolution >',(IFWIN#'+'),' i')&cd bind '') "1
glzqcolormode=: (('"',libjqt,'" glzqcolormode >',(IFWIN#'+'),' i')&cd bind '') "1
glzqduplexmode=: (('"',libjqt,'" glzqduplexmode >',(IFWIN#'+'),' i')&cd bind '') "1
glzqorientation=: (('"',libjqt,'" glzqorientation >',(IFWIN#'+'),' i')&cd bind '') "1
glzqoutputformat=: (('"',libjqt,'" glzqoutputformat >',(IFWIN#'+'),' i')&cd bind '') "1
glzqpageorder=: (('"',libjqt,'" glzqpageorder >',(IFWIN#'+'),' i')&cd bind '') "1
glzqpapersize=: (('"',libjqt,'" glzqpapersize >',(IFWIN#'+'),' i')&cd bind '') "1
glzqpapersource=: (('"',libjqt,'" glzqpapersource >',(IFWIN#'+'),' i')&cd bind '') "1

NB. =========================================================
glzqwh=: 3 : 0"1
wh=. 2#1.1-1.1
chkgl2 cdrc=. ('"',libjqt,'" glzqwh  ',(IFWIN#'+'),' i *f i') cd wh;y
1{::cdrc
)

NB. =========================================================
glzqmargins=: 3 : 0"1
ltrb=. 4#1.1-1.1
chkgl2 cdrc=. ('"',libjqt,'" glzqmargins  ',(IFWIN#'+'),' i *f i') cd ltrb;y
1{::cdrc
)

NB. =========================================================
NB. TODO
glzqextent=: 3 : 0"1
wh=. 2#2-2
chkgl2 cdrc=. ('"',libjqt,'" glzqextent  ',(IFWIN#'+'),' i *c *i') cd (,y);wh
2{::cdrc
)

NB. =========================================================
NB. TODO
glzqextentw=: 3 : 0"1
y=. y,(LF~:{:y)#LF [ y=. ,y
w=. (+/LF=y)#2-2
chkgl2 cdrc=. ('"',libjqt,'" glzqextentw  ',(IFWIN#'+'),' i *c *i') cd y;w
2{::cdrc
)

NB. =========================================================
NB. font information: Height, Ascent, Descent, InternalLeading, ExternalLeading, AverageCharWidth, MaxCharWidth
NB. TODO
glzqtextmetrics=: 3 : 0"1
tm=. 7#2-2
chkgl2 cdrc=. ('"',libjqt,'" glzqtextmetrics  ',(IFWIN#'+'),' i *i') cd <tm
1{::cdrc
)

NB. enum Qprinter::ColorMode
QPrinter_Color=: 1     NB.   print in color if available, otherwise in grayscale.
QPrinter_GrayScale=: 0     NB.   print in grayscale, even on color printers.

NB. enum Qprinter::DuplexMode
QPrinter_DuplexNone=: 0     NB.   Single sided (simplex) printing only.
QPrinter_DuplexAuto=: 1     NB.   The printer's default setting is used to determine whether duplex printing is used.
QPrinter_DuplexLongSide=: 2     NB.   Both sides of each sheet of paper are used for printing. The paper is turned over its longest edge before the
QPrinter_DuplexShortSide=: 3     NB.   Both sides of each sheet of paper are used for printing. The paper is turned over its shortest edge before the second side is printed

NB. enum Qprinter::Orientation
QPrinter_Portrait=: 0     NB.   the page's height is greater than its width.
QPrinter_Landscape=: 1     NB.   the page's width is greater than its height.

NB. enum Qprinter::OutputFormat
QPrinter_NativeFormat=: 0     NB.   QPrinter will print output using a method defined by the platform it is running on. This mode is the default
QPrinter_PdfFormat=: 1     NB.   QPrinter will generate its output as a searchable PDF file. This mode is the default when printing to a
QPrinter_PostScriptFormat=: 2     NB.   QPrinter will generate its output as in the PostScript format.


NB. enum Qprinter::PageOrder
QPrinter_FirstPageFirst=: 0     NB.   the lowest-numbered page should be printed first.
QPrinter_LastPageFirst=: 1     NB.   the highest-numbered page should be printed first.

NB. enum Qprinter::PaperSize
QPrinter_A0=: 5     NB.   841 x 1189 mm
QPrinter_A1=: 6     NB.   594 x 841 mm
QPrinter_A2=: 7     NB.   420 x 594 mm
QPrinter_A3=: 8     NB.   297 x 420 mm
QPrinter_A4=: 0     NB.   210 x 297 mm, 8.26 x 11.69 inches
QPrinter_A5=: 9     NB.   148 x 210 mm
QPrinter_A6=: 10    NB.   105 x 148 mm
QPrinter_A7=: 11    NB.   74 x 105 mm
QPrinter_A8=: 12    NB.   52 x 74 mm
QPrinter_A9=: 13    NB.   37 x 52 mm
QPrinter_B0=: 14    NB.   1000 x 1414 mm
QPrinter_B1=: 15    NB.   707 x 1000 mm
QPrinter_B2=: 17    NB.   500 x 707 mm
QPrinter_B3=: 18    NB.   353 x 500 mm
QPrinter_B4=: 19    NB.   250 x 353 mm
QPrinter_B5=: 1     NB.   176 x 250 mm, 6.93 x 9.84 inches
QPrinter_B6=: 20    NB.   125 x 176 mm
QPrinter_B7=: 21    NB.   88 x 125 mm
QPrinter_B8=: 22    NB.   62 x 88 mm
QPrinter_B9=: 23    NB.   33 x 62 mm
QPrinter_B10=: 16    NB.   31 x 44 mm
QPrinter_C5E=: 24    NB.   163 x 229 mm
QPrinter_Comm10E=: 25    NB.   105 x 241 mm, U.S. Common 10 Envelope
QPrinter_DLE=: 26    NB.   110 x 220 mm
QPrinter_Executive=: 4     NB.   7.5 x 10 inches, 190.5 x 254 mm
QPrinter_Folio=: 27    NB.   210 x 330 mm
QPrinter_Ledger=: 28    NB.   431.8 x 279.4 mm
QPrinter_Legal=: 3     NB.   8.5 x 14 inches, 215.9 x 355.6 mm
QPrinter_Letter=: 2     NB.   8.5 x 11 inches, 215.9 x 279.4 mm
QPrinter_Tabloid=: 29    NB.   279.4 x 431.8 mm
QPrinter_Custom=: 30    NB.   Unknown, or a user defined size.

NB. enum Qprinter::PaperSource
NB. Warning: This is currently only implemented for Windows.
QPrinter_Auto=: 6
QPrinter_Cassette=: 11
QPrinter_Envelope=: 4
QPrinter_EnvelopeManual=: 5
QPrinter_FormSource=: 12
QPrinter_LargeCapacity=: 10
QPrinter_LargeFormat=: 9
QPrinter_Lower=: 1
QPrinter_MaxPageSource=: 13
QPrinter_Middle=: 2
QPrinter_Manual=: 3
QPrinter_OnlyOne=: 0
QPrinter_Tractor=: 7
QPrinter_SmallFormat=: 8

NB. enum Qprinter::Unit
QPrinter_Millimeter=: 0
QPrinter_Point=: 1
QPrinter_Inch=: 2
QPrinter_Pica=: 3
QPrinter_Didot=: 4
QPrinter_Cicero=: 5
QPrinter_DevicePixel=: 6

glzinitprinter ::0:''  NB. initialise session printer
