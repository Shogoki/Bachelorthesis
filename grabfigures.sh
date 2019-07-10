^grep '!\[' source/*.md | cut -d '{' -f2 | cut -d '}' -f1 >> source/96_abbildungsverz.md
