# Motivation
Mithilfe von überwachten Produkttests mit anschließendem Interview kann die Marketing wertvolle Erkenntnisse über den möglichen erfolg von neuen Produkten gewinnen. In der Regel werden die im Anschluss abgegebenen Bewertungen der Tester dafür ausgewertet. Auch wenn die daraus extrahierten Informationen bereits einen deutlichen Gewinn für das Marketing darstellen, bleiben viele Informationen, welche aus einer Überwachung der Emotionen des Testers während des eigentlichen Produkttests hervorgehen können liegen.
Daher werten bereits einige Unternehmen diese überwachten Produkttests in einem zeitaufwändigen Prozess aus um die emotionelle Reaktion in bestimmten Situationen des Tests festzustellen. Um die dafür benötigten Aufwände zu minimieren ist es erstrebenswert diese zu automatisieren.

# Zielsetzung und Umfeld

## Umfeld der Thesis

Die Arbeit wird im Rahmen meiner beruflichen Tätigkeit bei der Evonik Industries AG geschrieben. 
Die Evonik Industries AG ist ein weiltweit führendes Unternehmen der Spezialchemie mit Sitz in Essen. Unter diesem Namen existiert Sie seit dem 12. September 2007. Es handelte sich dabei jedoch nicht um eine Neugründung sondern um eine Umfirmierung der RAG Beteiligungs AG.
Die Abteilung "Digital Labs" der internen IT ist verantwortlich für die Konzeptionisierung und Erstellung von Prototypen für neue Ideen und Anforderungen aus den operativen Segmenten im Rahmen der Digitalisierung des Konzerns. 
Eine solche Anforderung besteht für die Auswertung der Emotionen von Testern während Produkttests/-Interviews. Das operative Segment stellt die Videodaten zur Verfügung und möchte dann Anhand dieser die erkannten Emotionen in einer Zeitleiste bekommen.

## Zielsetzung

Das Ziel dieser Arbeit ist es einen Webservice zu entwickeln, welcher die Videodaten der Produkttests und/oder Interviews entgegen nimmt und eine Zeitleiste mit den anhand der Mimik erkannten Emotionen zurück gibt. Der Service soll als Prototyp entwickelt werden, um im Nachhinein von der operativen Evonik-IT weiterentwickelnt und betrieben zu werden.

Zur Realisierung eines solchen Service soll eine Antwort auf folgende Fragen liefern:

* Wie kann man menschliche Emotionen sinnvoll klassifizeren?
* Welche bereits verfügbaren Modelle können für diesen Anwendungsfall genutzt und entsprechend spezialiseirt werden?
* Woher bekommt man genug entsprechend gelabelte Trainingsdaten?
<!-- TODO: Weitere fragen -->

## Zielgruppe

Die primäre Zielgruppe für den Webservice ist die anfordernde Marketing Abteilung des Segmentes Nutrition & Care. Diese erstellen Videos von Produkttests und -Interviews welche dann an den neuen Webservice übermittelt werden sollen.

## Abgrenzung

Die folgenden Teile werden zwar in der Thesis kurz erwähnt können jedoch aufgrund der zur Verfügung stehenden Zeit nicht vollständig behandelt werden:

* Emotionserkennung aus anderen verfügbaren Daten als der Mimik, wie z.B. gesprochener Text & die verwndete entsprechende Tonlage
* Design und vollständige Implementierung eines Frontends für den Webservice

# Bedeutung

Die Bedeutung der Arbeit liegt primär in der Bereitstellung eines passenden Models, der benötigten Datenaufbereitung und Klassifizierung dieser. Das neuronale Netz soll genügend divers trainiert werden um die entsprechenden Emotionen bei verschiedenen Menschen mit einer hohen Trefferquote richtig zu erkennen.

# Methoden

# Zeitplan