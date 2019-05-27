# Einführung

Mithilfe überwachter Produkttests können zwar schon Heute wertvolle Informationen über den zu Erwartenden Erfolg eines Produktes getroffen werden, jedoch werden die zur Verfügung gestellten Informationen nur ineffizient genutzt. Eine Erfassung der Emotionen, welche der Tester während des Interviews empfindet könnte hier wichtige Erkenntnisse liefern. Dieser Ansatz ist nicht neu, in der Regel werden diese bereits vom Interviewer erfasst. Dieser ordnet subjektiv, aufgrund seiner Erfahrung und empathischen Fähigkeiten, das Verhalten seines Gegenübers bestimmten Emotionen zu. Im Sinne der Vergleichbarkeit, sowie der Effizienz ist es jedoch wünschenswert diese Analysen automatisiert durchzuführen. 
Die Idee des empathischen Computers fasziniert viele Forscher. Die Arbeiten auf diesem Gebiet erschließen das noch realtiv junge Gebiegt des **Affective Computings**, welches sich mit der Erkennung von Benutzergefühlen und anderen Stimuli, sowie adäquaten Reaktionen auf diese beschäftigt.[@Schuller2006]
Die machinelle Erkennung von Emotionen ist jedoch aufgrund der Komplexität und der Individualität des menschlichen Verhaltens nicht wirklich trivial, da die Körpersprache bei vielen Menschen zum Beispiel auch kulturell bedingt unterschiedlich ausfallen kann. Aber auch innerhalb eines Kulturkreises erweist sich eine solide Klassifizierung von Emotionen als schwierig. 
Um überhaupt eine solche Klassifizierung zu ermöglichen, müssen Emotionen zunächst einmal in Klassen eingeteilt werden und entsprechende Merkmale gefunden werden. Im Bereich der Bilderkennung, bzw. Klassifizierung haben sich in den letzten Jahren künstliche neuronale Netze als führende Technologie durchgesetzt. Diese Arbeit untersucht das Problem der Emotionserkennung mithilfe dieser, genauer gesagt **gefalteten neuronalen Netze** (engl.: convolutional neural network).

## Eingrenzung

Die Erkennung von Emotionen kann mithilfe von Videodaten auf vielfältige Wege geschehen. So bietet ein Video in der Regel Ton-Daten und Bild-Daten. Aus den gewonnen Ton-Daten können Emotionen anhand von spezifischen Stimmlagen erkannt werden. Des Weiteren kann das gesprochene mit **Sprache-zu-Text** (engl. Speech-to-Text) Systemen in Text umgewandelt und kontextuell ausgewertet werden. 
Aus den Bild-Daten können Emotionen aus Gestik, sowie aus Mimik abgeleitet werden. Aufgrund des begrenzten Zeitraums wird lediglich Letzteres ausführlich in dieser Arbeit behandelt.

<!--
Die automatisierte Erkennung von Emotionen aus Bildern ist kein grundlegend neues Thema. So wurde es zum Beispiel schon untersucht.... (TODO REF). Als Abgrenzung zu .... wird in dieser Arbeit  -->

## Ziel der Untersuchung

Das Ziel der Arbeit ist es, eine neuronales Netzwerk zu entwerfen und zu trainieren, welches Emotionen von Testern anhand der Mimik "erkennen" kann. Dieses soll von einem Webservice genutzt werden, welcher Videodaten von Probanden entgegen nimmt und eine Zeitleiste mit Emotionen zurück gibt. Der Service soll als Prototyp entwickelt werden.

Zur Realisierung eines solchen Service soll diese Arbeit eine Antwort auf folgende Fragen liefern:

* Wie kann man menschliche Emotionen sinnvoll klassifizieren?
* Welche Modelle können für diesen Anwendungsfall genutzt und entsprechend spezialisiert werden?
* Woher bekommt man entsprechende Trainingsdaten

<!--## Stand der Forschung -->

## Aufbau der Arbeit


