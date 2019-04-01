# Motivation
Mit Hilfe von überwachten Produkttests und anschließenden Interviews kann die Marketingabteilung schon Heute wertvolle Erkenntnisse über den möglichen Erfolg oder Misserfolg von neuen Produkten gewinnen. In der Regel werden die im Anschluss abgegebenen Bewertungen der Tester dafür ausgewertet.
Auch wenn die daraus extrahierten Informationen bereits einen deutlichen Gewinn für das Marketing darstellen, bleiben viele Informationen, welche aus einer Feststellung der Emotionen des Testers während des Produkttests hervorgehen können, unbeachtet.
Daher werten bereits einige Unternehmen diese überwachten Produkttests in einem zeitaufwändigen Prozess aus, um die emotionelle Reaktion in bestimmten Situationen des Tests festzustellen. Um die dafür benötigten Aufwände zu minimieren, ist es erstrebenswert diese zu automatisieren.

# Zielsetzung und Umfeld

## Umfeld der Thesis

Die Arbeit wird im Rahmen meiner beruflichen Tätigkeit bei der Evonik Industries AG durchgeführt.
Die Evonik Industries AG ist ein weltweit führendes Unternehmen der Spezialchemie mit Sitz in Essen. Unter diesem Namen existiert sie seit dem 12. September 2007. Es handelte sich dabei jedoch nicht um eine Neugründung, sondern um eine Umfirmierung der RAG Beteiligungs AG.
Die Abteilung "Digital Labs" der internen IT ist verantwortlich für die Konzeptionisierung und Erstellung von Prototypen. Diese sollen neue Ideen und Anforderungen aus den operativen Segmenten im Rahmen der Digitalisierung des Konzerns umsetzen.
Eine solche Anforderung besteht auch für die Auswertung der Emotionen während Produkttests/-interviews. Das operative Segment stellt entsprechende Videodaten zur Verfügung und möchte anhand dieser eine Zeitleiste der erkannten Emotionen erhalten.

## Zielsetzung

Das Ziel der Thesis ist es, eine neuronales Netzwerk zu entwickeln und zu trainieren, welches Emotionen von Testern anhand der Mimik "erkennen" kann. Dieses soll von einem Webservice genutzt werden, welcher die Videodaten der Produkttests und/oder Interviews entgegen nimmt und eine Zeitleiste mit Emotionen zurück gibt. Der Service soll als Prototyp entwickelt werden, um in Folge von der operativen Evonik-IT weiterentwickelt und betrieben zu werden.

Zur Realisierung eines solchen Service soll diese Arbeit eine Antwort auf folgende Fragen liefern:

* Wie kann man menschliche Emotionen sinnvoll klassifizieren?
* Welche bereits verfügbaren Modelle können für diesen Anwendungsfall genutzt und entsprechend spezialisiert werden?
* Wie kommt man bereits entsprechend klassifizierte Trainingsdaten?

## Zielgruppe

Die primäre Zielgruppe ist die anfordernde Marketingabteilung des Segmentes Nutrition & Care. Aber auch die "Digital Labs" selbst können von diesem profitieren um ihn als Basis für andere Projekte zu verwenden.

## Abgrenzung

Die folgenden Themen werden in der Thesis zwar kurz erwähnt, können jedoch aufgrund der zur Verfügung stehenden Zeit nicht vollständig behandelt werden:

* Emotionserkennung aus anderen verfügbaren Daten als der Mimik, wie z.B. gesprochener Text & die verwendete entsprechende Tonlage
* Design und vollständige Implementierung eines Frontends für den Webservice

# Bedeutung

Die Bedeutung der Arbeit liegt primär in der Bereitstellung eines passenden Modells, der benötigten Datenaufbereitung und Klassifizierung dieser. Das neuronale Netz soll mit genügend und ausreichend diversen Daten trainiert werden, um die entsprechenden Emotionen bei verschiedenen Menschen mit einer hohen Trefferquote richtig zu erkennen.

# Methodik

Im theoretischen Teil, welcher als Basis für die Implementierung des Prototypen dient, soll auf der Grundlage von einschlägiger Fachliteratur die sinnvolle Klassifizierung von Emotionen und der entsprechende Aufbau eines passenden neuronalen Netzwerkes herausgearbeitet werden.
Anschließend sollen die erworbenen Erkenntnisse mittels Prototyping bei einer ersten Implementierung & Evaluierung eines Emotions-Klassifizierungs-Service weiter vertieft werden.

# Vorläufiger Zeitplan

* Planungsphase
	* 01.04.2019 - Abgabe des Exposé´s
	* 03.04.2019 - Besprechung und ggf. Anpassung des Exposé´s
	* bis 14.04.2019: Anmeldung der Bachelor-Thesis
* Vorbereitungsphase (02.05.2019 - ca. 16.05.2019 )
	* Literatur-Recherche
	* Formulierung des Sachverhaltes & Forschungsfragen
	* Abstimmung mit dem Betreuer
* Strukturierungsphase (16.05.2019 - ca. 03.06.2019)
	* Strukturierung der Ergebnisse
	* Erstellung der Gliederung
	* Abstimmung mit dem Betreuer
* Schreibphase (03.06.2019 - ca. 28.06.2019)
	* Erstellen der Rohfassung der Thesis
	* Erstellen des Literaturverzeichnisses
	* Überprüfung der Rohfassung auf Vollständigkeit
	* Überprüfung der Rechtschreibung & Formulierungen
* Abschlussphase (28.06.2019 - ca.01.08.2019)
	* Korrekturlesen
	* Überarbeitung des Layout
	* Drucken & Binden
	* Abgabe der Bachelor Thesis


# Vorläufige Gliederung


* Abbildungsverzeichnis
* Abkürzungsverzeichnis
* Glossar
1. Einführung
	1. Eingrenzung 
	2. Ziel der Untersuchung 
	3. Stand der Forschung 
	4. Fragestellung
2. Theoretischer Teil
	1. Einordnung der Daten
		1. Beschaffenheit der Daten
		2. Vorbereitung der Daten
		3. Klassifizierung
			1. Unterteilung der Emotionen
			2. Zuordnung der Daten
	2. künstliche neuronale Netze
		1. Übersicht & Definition
		2. Deep Learning
		3. Convolutional Neural Network
		4. Trainingsmethoden
		5. Gesichtserkennung
	3. aktuell bewährte Modelle 
3. Methodik
	1. Prototyping
		1. Aufbau verschiedener neuronaler Netze
		2. Trainieren der Modelle
		3. Bereitstellung als Webservice
	2. Experiment
		1. Verifizieren mit Testdaten
		2. Untersuchung der Genauigkeit
4. Ergebnisse
5. Fazit
* Literaturverzeichnis