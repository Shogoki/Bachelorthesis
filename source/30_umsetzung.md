\clearpage

# Umsetzung & Evaluierung

In diesem Teil wird die Umsetzung der Arbeit eingegangen. im ersten Abschnitt wird auf die Art und Einteilung der zu klassifizierenden Daten eingegangen. Im Weiteren Verlauf wird auf die Beschaffung Der verschiedenen Datensätze eingegangen mit welchen gearbeitet werden soll. Anschließend wird auf diverse Verfahren der Datenpräparation wie Vorbereitung, Generalisierung und Datenvervielfältigung eingegangen. Letzen Endes werden verschiedene neuronale Netze entworfen, trainiert und getestet, bis dann im letzten Abschnitt auf die Realisierung des zu erstellenden Webservice eingegangen wird.

## Beschaffenheit und Einteilung der Daten

Wie bereits beschrieben sollen für den späteren Webservice Videodaten als Eingabe dienen. Das neuronale Netz wird jedoch nicht direkt die Videodaten, sondern aus dem Video extrahierte Einzelbilder als Eingabe erhalten. Diese Eingabedaten werden anhand später beschriebener Verfahren noch weiter vorbereitet und optimiert.
Die Einzelbilder sollen als *Array* von Pixelwerten eines Graustufenbildes an das neuronale Netzwerk übergeben werden (siehe Datenpräparation).
Die Bilder sollen in 7 verschiedene Klassen. Die verschiedenen Emotionen bilden den Zielklassenvektor $Z$, es gilt also $|Z| = 7$. Jedes Bild welches der selben Emotion des *FACS* entspricht ist Mitglied der selben Klasse aus $Z$.

## Datensätze

Ein Teil der Arbeit bestand darin geeignete Datensätze für das Training und die Evaluierung des neuronalen Netzes zu finden und diese später in entsprechende Trainings, Evaluierungs und Test-Datensätze für das neuronale Netzwerk zu unterteilen.

### Beschaffung der Datensätze

Dazu sollen 2 verschiedene Ansätze unterschieden werden. Zum einen die Beschaffung eines vorhandenen freien Datensatzes von Gesichtsbildern inklusive der Zuordnung zu einer der entsprechenden *FACS* Emotionen, sowie zum anderen die Generierung von eignen Daten mithilfe eines Webservice und freiwillgen Probanden. 
In dieser Arbeit wurden beide Ansätze in Kombination verwendet. Es wurden also 2 verschiedene Datenquellen verwendet, was bei der Aufteilung der Datensätze noch eine wichtige Rolle spielt (siehe Einteilung der Datensätze<!--TODO: this may changes -->). 

#### FER+

Als großer frei Verfügbarer Datensatz wurde der *Facial Expression Recognition+* (FER+)[@Barsoum2016] Datensatz verwendet. Bei den Eingangsaten des *FER+* handelt es sich um die selben Bilder, wie auch beim *FER2013*, welcher Teil der International Conference for Machine Learning (ICML) Challenge 2013 war, und danach der Öffentlichkeit zur Verfügung gestellt wurde. Bei FER+ wurden jedoch alle *Label*, mithilfe von *Crowdsourcing* neu erstellt, um eine bessere Datenqualität zu erreichen. (Vergleich [@Barsoum2016]). Der Datensatz besteht aus 34034 NxN Graustufen Bilder von Gesichtern. Jedes dieser Bilder von je 10 Taggern mithilfe von *Crowdsourcing* bewertet. Der Datensatz enthält für jede Klasse (Emotionen des *FACS* und "kein Gesicht") die Anzahl an Taggern, welche das Bild entsprechend bewertet haben.
Ein Beispiel für ein einzelnes Datum des Datensatzes ist in Abbildung \ref{single_ferplus} zu sehen

TODO: Abbildung FER+ Single row image

Das Team von Microsoft Research[@Barsoum2016] beschreibt mehrere Variationen, wie die mehrfach *gelabelten* Daten verwendbar sind. In dieser Arbeit wird jedoch ausschließlich der einfache Mehrheits-Ansatz verfolgt. Es wird also jedes Bild der Klasse zugeordnet, welche die meisten Stimmen erhalten hat.

#### selbsterstellte Daten

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