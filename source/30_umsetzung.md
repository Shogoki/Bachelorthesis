\clearpage

# Umsetzung & Evaluierung

In diesem Teil wird die Umsetzung der Arbeit eingegangen. im ersten Abschnitt wird auf die Art und Einteilung der zu klassifizierenden Daten eingegangen. Im Weiteren Verlauf wird auf die Beschaffung Der verschiedenen Datensätze eingegangen mit welchen gearbeitet werden soll. Anschließend wird auf diverse Verfahren der Datenpräparation wie Vorbereitung, Generalisierung und Datenvervielfältigung eingegangen. Letzen Endes werden verschiedene neuronale Netze entworfen, trainiert und getestet, bis dann im letzten Abschnitt auf die Realisierung des zu erstellenden Webservice eingegangen wird.

## Beschaffenheit und Einteilung der Daten

Wie bereits beschrieben sollen für den späteren Webservice Videodaten als Eingabe dienen. Das neuronale Netz wird jedoch nicht direkt die Videodaten, sondern aus dem Video extrahierte Einzelbilder als Eingabe erhalten. Diese Eingabedaten werden anhand später beschriebener Verfahren noch weiter vorbereitet und optimiert. 
Die Bilder sollen in 7 verschiedene Klassen. Die verschiedenen Emotionen bilden den Zielklassenvektor $\hat{Y}$, es gilt also $|\hat{Y}| = 7$. Jedes Bild welches der selben Emotion des *FACS* entspricht ist Mitglied der selben Klasse aus $\hat{Y}$.

## Datensätze

### Beschaffung der Datensätze

### Einteilung der Datensätze (Train/DEV_train/DEV/test)

## Datenpräparation

### Vorverarbeitung

### Generalisierung

### Datenmehrung

## Entwurf neuronaler Netze

### Topoligien

### Trainingsmethoden


## Evaluierung neuronaler Netze?

### abschnitt 1

### abschnitt 2

## Entwicklung eines Webservice




<!-- expose:
3. Umsetzung & Evaluierung
	1. Prototyping
		1. Aufbau verschiedener neuronaler Netze
		2. Trainieren der Modelle
		3. Bereitstellung als Webservice
	2. Experiment
		1. Verifizieren mit Testdaten
		2. Untersuchung der Genauigkeit
	3. Ergebnisse
>