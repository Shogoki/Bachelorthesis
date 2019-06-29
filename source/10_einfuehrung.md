
\mainmatter

# Einführung

Mithilfe überwachter Produkttests können zwar schon Heute wertvolle Informationen über den zu erwartenden Erfolg eines Produktes getroffen werden, jedoch werden die zur Verfügung gestellten Informationen nur ineffizient genutzt. Eine Erfassung der Emotionen, welche der Tester während des Interviews empfindet, könnte hier wichtige Erkenntnisse liefern. Dieser Ansatz ist nicht neu, in der Regel werden diese bereits vom Interviewer erfasst. Dieser ordnet subjektiv, aufgrund seiner Erfahrung und empathischen Fähigkeiten, das Verhalten seines Gegenübers bestimmten Emotionen zu. Im Sinne der Vergleichbarkeit, sowie der Effizienz ist es jedoch wünschenswert diese Analysen automatisiert durchzuführen.
Die Idee des empathischen Computers fasziniert viele Forscher. Die Arbeiten auf diesem Gebiet erschließen das noch relativ junge Gebiet des **Affective Computings**, welches sich mit der Erkennung von Benutzergefühlen und anderen Stimuli, sowie adäquaten Reaktionen auf diese beschäftigt.[@Schuller2006]

Die maschinelle Erkennung von Emotionen ist jedoch aufgrund der Komplexität und der Individualität des menschlichen Verhaltens nicht wirklich trivial, da die Körpersprache bei vielen Menschen, zum Beispiel auch kulturell bedingt, unterschiedlich ausfallen kann. Aber auch innerhalb eines Kulturkreises erweist sich eine solide Klassifizierung von Emotionen als schwierig.

Um überhaupt eine solche Klassifizierung zu ermöglichen, müssen Emotionen zunächst in Klassen eingeteilt werden und entsprechende Merkmale gefunden werden. Im Bereich der Bilderkennung, bzw. Klassifizierung haben sich in den letzten Jahren künstliche neuronale Netze als führende Technologie durchgesetzt. Diese Arbeit untersucht das Problem der Emotionserkennung mithilfe dieser Netze, genauer gesagt mit **gefalteten neuronalen Netze** (engl.: convolutional neural network).

## Eingrenzung

Die Erkennung von Emotionen kann mithilfe von Videodaten auf vielfältige Wege geschehen. So bietet ein Video in der Regel Ton-Daten und Bild-Daten. Aus den gewonnen Ton-Daten können Emotionen anhand von spezifischen Stimmlagen erkannt werden. Des Weiteren kann das Gesprochene mit **Sprache-zu-Text** (engl. Speech-to-Text) Systemen in Text umgewandelt und kontextuell ausgewertet werden.
Aus den Bild-Daten können Emotionen aus Gestik, sowie aus Mimik abgeleitet werden. Aufgrund des begrenzten Zeitraums wird lediglich Letzteres ausführlich in dieser Arbeit behandelt.

<!--
Die automatisierte Erkennung von Emotionen aus Bildern ist kein grundlegend neues Thema. So wurde es zum Beispiel schon untersucht.... (TODO REF). Als Abgrenzung zu .... wird in dieser Arbeit  -->

## Ziel der Untersuchung

Das Ziel der Arbeit ist es, ein neuronales Netzwerk zu entwerfen und zu trainieren, welches Emotionen von Testern anhand der Mimik "erkennen" kann. Dieses soll von einem Webservice genutzt werden, welcher Videodaten von Probanden entgegen nimmt und eine Zeitleiste mit Emotionen zurück gibt. Der Service soll als Prototyp entwickelt werden.

Zur Realisierung eines solchen Service soll diese Arbeit eine Antwort auf folgende Fragen liefern:

* Wie kann man menschliche Emotionen sinnvoll klassifizieren?
* Welche Modelle können für diesen Anwendungsfall genutzt und entsprechend spezialisiert werden?
* Woher bekommt man entsprechende Trainingsdaten?

<!--## Stand der Forschung -->

## Aufbau der Arbeit

Die Arbeit gliedert sich in drei Teilbereiche. Zuerst werden die im Rahmen einer Literatur-Recherche gewonnenen, theoretischen Grundlagen zur Kategorisierung von Emotionen, sowie künstlichen neuronalen Netzen und der Klassifizierung mit deren Hilfe erläutert.
<!-- TODO: anders schreiben? -->

Im darauf folgenden Teil wird auf Basis der Erkenntnisse des ersten Teils der Entwurf des künstlichen neuronalen Netzes, sowie die Schritte zur Datenbeschaffung und Vermehrung beschrieben. Ferner wird auch auf die gewählte Trainingsmethode, sowie die Implementierung des Webservice eingegangen.

Der dritte Teil befasst sich mit der Evaluierung des entworfenen Klassifizierers anhand einer tiefer gehenden Analyse <!--(TODO: empirisch?) -->. Ein Test des Webservices mit Videodaten von verschiedenen Personen wird durchgeführt, um die ausreichende Generalisierung des neuronalen Netzes zu überprüfen. <!-- TODO: Kapitel ref -->