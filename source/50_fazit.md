\clearpage

# Fazit

Bei der automatisierten Erkennung von Emotionen gibt es aktuell diverse Ansätze. Im Rahmen der gesichtsbasierten Emotionserkennung, in Form von Bildern, gibt es aktuell noch kaum dokumentierte Netzstrukturen, welche mit einer eher geringen Anzahl an Trainingsdaten auskommen.
Die Leistung der bestehenden Lösungen ist stark abhängig von der Anzahl der gewählten Klassen, was auf die Einteilung der Emotionen zurückzuführen ist.

Zu Beginn dieser Arbeit wurden zunächst verschiedene Ansätze für die Einteilung von Emotionen kurz betrachtet. Im Gebiet der Psychologie ist die Einteilung nach dem *FACS* weit verbreitet, weshalb diese in dieser Arbeit gewählt wurde. Eine Möglichkeit wäre die Einteilung in weniger Emotionen, was zu besseren Ergebnissen führen könnte. <!-- Bernhard TODO: Nicht so ne gute Möglichkeit? Anderrs formulieren-->
<!-- In anderen Arbeiten zum Thema Emotionserkennung wurde die Einteilung in weniger Emotionen vorgenommen (Vergleich XY), was zu besseren Ergebnissen geführt hat. -->

Im Anschluss wurden frei verfügbare Trainingsdaten gesucht und beschafft. Dabei wurde erkenntlich, dass im Bereich der Emotionserkennung bisher sehr wenige frei verfügbare Datensätze in ausreichender Größe und Qualität zu finden sind. Zusätzlich wurde ein geringer Anteil an Daten selbst erstellt, um die Leistung des Netzes besser evaluieren zu können. Hierbei zeigte sich bereits, dass nicht eindeutige Emotionen, die selbst Menschen schwer interpretieren können, eine weitere besondere Herausforderung für die Maschine sind.
<!--
Eine weitere besondere Herausforderung für die Maschine sind nicht eindeutige Emotionen, die selbst Menschen schwer interpretieren können.
 -->

<!-- NICEMAKING -->
\clearpage

Bei der Verwendung von zwei verschiedenen Bild-Datensätzen wurde in der Arbeit auch detailliert auf die Vorverarbeitung von solchen Daten eingegangen. Hier wurden verschiedene Möglichkeiten zur Transformation der Daten in ein trainier- bzw. lernbares Format, sowie zur Vervielfachung betrachtet.

Für diese Daten wurde anschließend ein gut funktionierendes Netzwerk-Modell mit möglichst optimalen Parametern gesucht. Hierbei wurden vorerst zwei verschiedene Netzwerkarchitekturen verglichen, wovon das Erste aber nach einigen Testläufen verworfen wurde. Daraufhin wurde sich noch mit der *Transfer Learning* Variante beschäftigt. Dazu ist eine Vielzahl an möglichen Parametern evaluiert worden, um ein möglichst gutes Modell zu finden. Hierbei wurde schnell erkannt, dass das Modell leicht zur Überanpassung auf den Trainingsdatensatz neigt, was im Rahmen der Arbeit zwar nicht vollständig behoben werden konnte, dafür aber Maßnahmen abgeleitet werden konnten, die für eine zukünftige Optimierung verwendet werden können.
Das beste gefundene Modell wurde weiterführend analysiert und festgestellt, dass einige Emotionen besonders schlecht gegenüber anderen erkannt wurden.

 Abschließend wurde ein Webservice erstellt, welcher das gefundene Modell zur Klassifizierung nutzt. Er dient als Benutzerschnittstelle und kümmert sich um die Extraktion von Einzelbildern aus einem empfangenen Video, welche er zum Klassifizierer sendet, um die jeweilige Emotion zu erkennen.

Aufgrund der begrenzten Bearbeitungszeit und -mittel konnte im Rahmen dieser Arbeit das Modell nicht weiter optimiert werden. Es wurden jedoch wichtige Erkenntnisse erlangt, welche helfen das neuronale Netzwerk in Zukunft weiter zu entwickeln. 

**Tiefere Analyse der Trainingsdaten:** Die Trainingsdaten sollten weitergehend unter dem Gesichtspunkt der schlecht erkannten Emotionen analysiert werden. Unter Umständen gibt es für einige Emotionen zu wenig oder unzureichend gute Daten. Des weiteren sollten die Trainingsdaten auf die Unterschiede zum Entwicklungs- und Test Datensatz untersucht werden, da hier eine weitere Diskrepanz besteht.

**Weitergehende Regularisierung:** Um der Überanpassung entgegen zu wirken bietet es sich an, das Netzwerk mit weiteren Regularisierungs-Verfahren auszubauen.

**Beschaffung von weiteren Trainingsdaten** Eine weitere Idee zur Verbesserung des Netzwerkes ist es, mehr Trainingsdaten zu beschaffen. Dazu könnte zum Beispiel der genutzte Webservice zum Selbsterstellen von Daten weiter verbreitet und verwendet werden, um ausreichend eigene Trainingsdaten zu beschaffen.

Die Implementierung des Webservice wurde im Rahmen dieser Arbeit von Anfang an lediglich als Prototyp geplant. Dieser könnte zum einen um eine anschauliche Benutzeroberfläche erweitert und zum anderen in der Funktionalität ausgebaut werden.
So wären, nachdem die gesichtsbasierte Emotionserkennung ausreichend gut funktioniert, weitere Erweiterungen auf die Erkennung der Emotionen anhand der Tonlage oder der Sprache (*Text2speech*) der Audiodaten der Videos möglich.
