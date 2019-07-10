
\mainmatter

# Einführung

Mithilfe überwachter Produkttests können schon Heute wertvolle Informationen über den zu erwartenden Erfolg eines Produktes getroffen werden, jedoch werden die zur Verfügung gestellten Informationen nur ineffizient genutzt. Eine Erfassung der Emotionen, welche der Tester während des Interviews empfindet, könnte hier wichtige Erkenntnisse liefern. So könnte zum Beispiel die eine Reaktion wie Ekel bei der Anwendung eines Produktes gedeutet werden. Dieser Ansatz ist nicht neu, in der Regel werden diese bereits vom Interviewer erfasst.<!-- Bernhard TODO: Verweis? --> Dieser ordnet subjektiv, aufgrund seiner Erfahrung und empathischen Fähigkeiten, das Verhalten seines Gegenübers bestimmten Emotionen zu. Im Sinne der Vergleichbarkeit, sowie der Effizienz ist es jedoch wünschenswert diese Analysen automatisiert durchzuführen.
Die Idee des empathischen Computers fasziniert viele Forscher. Die Arbeiten auf diesem Gebiet erschließen das noch relativ junge Gebiet des **Affective Computings**, dass sich mit der Erkennung von Benutzergefühlen und anderen Stimuli, sowie adäquaten Reaktionen auf diese beschäftigt[@Schuller2006].

Die maschinelle Erkennung von Emotionen ist aufgrund der Komplexität und der Individualität des menschlichen Verhaltens nicht trivial, da die Körpersprache bei vielen Menschen, zum Beispiel auch kulturell bedingt, unterschiedlich ausfallen kann. Aber auch innerhalb eines Kulturkreises erweist sich eine solide Klassifizierung von Emotionen als schwierig.

Um eine solche Klassifizierung zu ermöglichen, müssen Emotionen zunächst in Klassen eingeteilt und entsprechende Merkmale gefunden werden. Im Bereich der Bilderkennung, bzw. Klassifizierung, also der Einteilung von Bildern in eine vordefinierte Anzahl an Klassen, haben sich in den letzten Jahren künstliche neuronale Netze als führende Technologie durchgesetzt. <!-- Bernhard TODO: Verweis? --> Diese Arbeit untersucht das Problem der Emotionserkennung mithilfe dieser Netze, genauer gesagt mit **gefalteten neuronalen Netze** (engl.: convolutional neural network).

## Eingrenzung

Die Erkennung von Emotionen kann mithilfe von Videodaten auf vielfältige Wege geschehen. So bietet ein Video in der Regel Ton-Daten und Bild-Daten. Aus den gewonnen Ton-Daten können Emotionen anhand von spezifischen Stimmlagen erkannt werden. Des Weiteren kann das Gesprochene mit **Sprache-zu-Text** (engl. Speech-to-Text) Systemen in Text umgewandelt und kontextuell ausgewertet werden.
Aus den Bild-Daten können Emotionen aus Gestik, sowie aus Mimik abgeleitet werden. Aufgrund des begrenzten Zeitraums wird lediglich Letzteres ausführlich in dieser Arbeit behandelt.

<!--
Die automatisierte Erkennung von Emotionen aus Bildern ist kein grundlegend neues Thema. So wurde es zum Beispiel schon untersucht.... (TODO REF). Als Abgrenzung zu .... wird in dieser Arbeit  -->

## Ziel der Untersuchung

Das Ziel der Arbeit ist, ein neuronales Netzwerk zu entwerfen und zu trainieren, dass Emotionen von Testern anhand der Mimik "erkennen", bzw. klassifizieren kann. Dieses soll von einem, zu erstellenden Webservice genutzt werden, der Videodaten von Probanden entgegen nimmt und eine Zeitleiste mit Emotionen berechnet und zurückliefert. Der Service soll als Prototyp entwickelt werden.

Zur Realisierung eines solchen Service soll diese Arbeit eine Antwort auf folgende Fragen liefern:

* Wie können menschliche Emotionen sinnvoll klassifiziert werden?
* Welche Datenquellen können für die Anforderungen verwendet werden?
* Wie müssen die gegebenen Daten vorbereitet, bzw.präpariert werden, um ein neuronales Netz zu verwenden?
* Welche Modelle <!-- Bernhard TODO: Ref. Modelle? --> können für diesen Anwendungsfall genutzt und entsprechend spezialisiert werden?

<!--## Stand der Forschung -->

## Aufbau der Arbeit

Die Arbeit gliedert sich in drei Teilbereiche. Zu Beginn werden in Kapitel \ref{chapter_grundlagen} die, im Rahmen einer Literatur-Recherche gewonnenen, theoretischen Grundlagen zur Kategorisierung von Emotionen, sowie künstlichen neuronalen Netzen und der Klassifizierung mit deren Hilfe erläutert.
<!-- TODO: anders schreiben? -->

Daraufhin wird in Kapitel \ref{chapter_umsetzung} auf Basis der Erkenntnisse des ersten Teils der Entwurf des künstlichen neuronalen Netzes, sowie die Schritte zur Datenbeschaffung und Vermehrung beschrieben. Ferner wird auch auf die Suche nach den besten Hyperparametern, sowie die Implementierung des Webservice eingegangen.

Kapitel \ref{chapter_evaluierung} befasst sich mit der weiterführenden Evaluierung des entworfenen Klassifizierers. Ein Test des Webservices mit Videodaten von verschiedenen Personen wird durchgeführt, um die ausreichende Generalisierung des neuronalen Netzes zu überprüfen. <!-- TODO: Kapitel ref -->
