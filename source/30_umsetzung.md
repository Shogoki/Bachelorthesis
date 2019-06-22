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

Zum selbst erstellen von Daten wurde im Rahmen der Arbeit eine einfache Website erstellt, welche mithilfe von *WebRTC* Zugriff auf die Kamera bekommt. Auf dieser Website haben freiwillige Probanden, und auch der Autor dieser Arbeit, die Möglichkeit nacheinander für jede Emotion des *FACS* ein 15 sekündiges Video aufzunehmen. Dieses wird anschließend direkt auf dem Server gespeichert. Die Webseite ist in Abbildung \ref{webrtc_screenshot} zu sehen.

TODO: Screenshot Website

Mithilfe dieser Webseite wurden insgesamt 25 Sätze an 15 sekündigen Videos von 10 verschiedenen freiwilligen Probanden (den Autor dieser Arbeit eingeschlossen) gesammelt. Aus diesen Videos wurde anschließend mit Hilfe eines python-Skripts (siehe TODO: Listing) pro Sekunde ein Einzelbild extrahiert, und mit dem Namen der entsprechenden Klasse abgespeichert. Somit wurden also $25 * 15 = 375$ Einzelbilder pro Klasse generiert.

TODO: Code Listing


Die Einzelbilder wurden anschließend vom Autor manuell auf Korrektheit, das heißt Zuordnung zur Klasse, geprüft. Dabei wurden insgesamt 100 Bilder wieder aussortiert (TODO: Verify numbers).


### Einteilung der Datensätze (Train/DEV_train/DEV/test)

Beim machinellen Lernen ist es üblich den vorhandenen Datensatz, bzw. die vorhandenen Datensätze in verschiedene Verwendungszwecke einzuteilen. Klassischerweise sprach man hier immer vom *train/test-Split*, also einer Aufteilung der Daten in einen Trainings- und einen Test-Datensatz. In modernen Projekten, welche sich mit maschinellen Lernen beschäftigen spricht man jedoch zumeist von einem *train/dev/test-split*. Die Daten werden also in einen Trainings-, einen Entwicklungs- und einen Test-Datensatz eingeteilt. Als Entwicklungs-Datensatz bezeichnet man, jene Daten, welche während der Entwicklung, also dem Anpassen bestimmter (Hyper-)Parameter, des neuronalen Netzes zur Evaluierung verwendet. Der Test-Datensatz ist in diesem Szenario ein Satz aus Daten, welches das neuronale Netz vor der Fertigstellung noch nicht "zu sehen" bekommen hat. Beim klassichen *train/test-split* ist der Test-Satz also eingetlich, das was wir Heute als Entwicklungs-Datensatz bezeichnen, und es gibt keinen wirklichen Test-Datensatz.
Bei der Wahl der Datenquellen, ist es wichtig, dass die Test-Daten möglichst ähnlich, zu den später erwarteten Eingangsdaten sind, und dass Entwicklungs- und Test-Datensatz aus der selben Quelle stammen sollten.

Für diese Arbeit bedeutet das, dass die Entwicklungs- und Test-Daten aus den selbstgenerierten Daten stammen, da diese Bereits von aufgenommenen Videos stammen, was den Zeildaten sehr nahe kommt.
Als Trainingsdaten wird entsprechend der *FER+* Datensatz verwendet.
EIn Problem bei einer solchen Aufteilung, wenn also die Trainingsdaten aus einem anderen Datensatz stammen als die Entwicklungs und Test-Daten ist, dass man gewisse Probleme, wie zum Beispiel eine Überanpassung manchmal nur schwer erkennen kann. Deshalb ist es in einem solchen Fall sinnvoll noch einen vierten Datensatz einzuführen, welcher aus der selben Quelle wie die Trainingsdaten stammt (hier *FER+*). Man spricht hier vom *dev_train* oder auch *bridge* Datensatz. Dieser wird im Prinzip analog zum Entwicklungsdatensatz behandelt, und dient zum testen der Parameter des neuronalen Netzwerkes, nach jeder Änderung. Anhand der Unterschiedlichen Ergebnisse für den *bridge* und den *dev* Datensatz kann man nun schnell, bestimmte Probleme des neuronalen Netzwerks erkennen.

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
-->