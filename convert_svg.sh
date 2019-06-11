#!/bin/bash
for f in source/figures/*.svg; do
	PDFNAME=$(echo $f | sed -En "s/\.svg$/\.pdf/p")
	echo Converting $f to $PDFNAME 
	inkscape $PWD/$f --export-pdf=$PWD/$PDFNAME
done #< $(ls ) 
#
