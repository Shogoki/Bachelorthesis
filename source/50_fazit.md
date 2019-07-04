\clearpage

# Fazit

Bei der automatisierten Erkennung von Emotionen gibt es aktuell diverse verschiedene Ansätze. Im Rahmen der gesichtsbasierten Emotionserkennung anhand von Bildern gibt es aktuell noch kaum dokumentierte Netzstrukturen, welche mit einer eher geringen Anzahl an Trainingsdaten auskommen.
Die Leistung der bestehenden Lösungen ist stark abhängig von der Anzahl der gewählten Klassen, was auf die Einteilung der Emotionen zurück zu führen ist. 
Zu Beginn dieser Arbeit wurden zunächst verschiedenen Ansätze für die Einteilung von Emotionen kurz betrachtet. Im Gebiet der Psychologie ist die Einteilung nach dem *FACS* weit verbreitet, weshalb diese in dieser Arbeit gewählt wurde. Eine Möglichkeit wäre die Einteilung in weniger Emotionen, was zu besseren Ergebnissen führen könnte.
<!-- In anderen Arbeiten zum Thema Emotionserkennung wurde die Einteilung in weniger Emotionen vorgenommen (Vergleich XY), was zu besseren Ergebnissen geführt hat. -->

Im Anschluss wurden frei verfügbare Trainingsdaten gesucht und Beschafft. Dabei wurde erkenntlich, dass im Bereich der Emotionserkennung bisher sehr wenige frei verfügbare Datensätze in ausreichender Größe und Qualität gut verwendbare zu finden sind. Zusätzlich wurde ein geringer Anteil an Daten selbst erstellt um die Leistung des Netzes besser evaluieren zu können. Hierbei zeigte sich bereits eine Schwierigkeit des Problems, dass auch Menschen einige Emotionen nicht eindeutig klassifizieren können. 

Bei der Arbeit mit den  zwei verschiedenen Bild-Datensätzen wurde in der Arbeit auch detailliert auf die Vorverarbeitung von solchen Daten eingegangen. Hier wurden verschiedene Möglichkeiten, zur Transformation der Daten in ein trainier- bzw. lernbares Format, sowie zur Vervielfachung betrachtet.

Für diese Daten wurde anschließend ein gut funktionierendes Netzwerk-Model, mit möglichst optimalen Parametern gesucht. Hierbei wurden vorerst zwei verschienden Netzwerkarchitekturen verglichen, wovon das erste aber nach einigen Testläufen verworfen wurde. Daraufhin wurde sich nurmehr noch mit der *Transfer Learning* Variante beschäftigt. Hierbei wurde eine Vielzahl an möglichen Parametern evaluiert um ein möglichst gutes Model zu finden. Hierbei wurde schnell erkannt, dass das Model leicht zur Überanpassung auf den Trainingsdatensatz neigt, was im Rahmen der Arbeit zwar nicht vollständig behoben werden konnte, dafür aber Maßnahmen abgeleitet werden konnte, welche für eine zukünftige Optimierung verwendet werden können.
Das beste gefundene Model wurde weiterführend analysiert und festgestellt, dass einige Emotionen besonders schlecht gegenüber anderen erkannt wurden.

 Abschließend wurde ein Webservice erstellt, welcher das gefundene Model zur Klassifizierung nutzt. Er dient als Benutzerschnittstelle und kümmert sich um die Extraktion von Einzelbildern aus einem empfangenen Video, welche er zum Klassifizierer sendet um die jeweilige Emotion zu erkennen.

Aufgrund der begrenzten Bearbeitungszeit und Mittel konnte im Rahmen dieser Arbeit, dass Model nicht viel weiter optimiert werden. Es wurden jedoch wichtige Erkenntnisse erlangt welche helfen das neuronale Netzwerk in Zukunft weiter zu entwickeln. 

**tiefere Analyse der Trainingsdaten** Die Trainingsdaten sollten weitergehend unter dem Gesichtspunkt der schlecht erkannten Emotionen analysiert werden. Unter Umständen gibt es für einige Emotionen zu wenig unter unzureichend gute Daten. Zum anderen sollten die Trainingsdaten auf die Unterschiede zum Entwicklungs- und Test Datensatz untersucht werden. Da hier eine weitere Diskrepanz besteht.

**weitergehende Regularisierung** Um der Überanpassung entgegen bietet es sich an Das Netzwerk noch weiter mit Regularisierungs-Verfahren auszubauen.

**Beschaffung von weiteren Trainingsdaten** Eine weitere Idee zur Verbesserung des Netzwerkes ist es mehr Trainingsdaten zu beschaffen. Dazu könnte zum Beispiel der genutzte Webservice zum selbsterstellen von Daten noch weiter verbreitet und verwendet werden um ausreichen eigene Trainingsdaten zu Beschaffen.

Die Implementierung des Webservice wurde im Rahmen dieser Arbeit von Anfang an, lediglich als Prototyp geplant. Dieser hat ebenfalls könnte zukünftig zum einen um eine anschauliche Benutzeroberfläche erweitert werden und zum anderen in der Funktionalität ausgebaut werden.
So wären, nachdem die Gesichtsbasierte Gesichtserkennung ausreichend gut funktioniert, weitere Erweiterungen auf die Erkennung der Emotionen anhand der Tonlage oder der Sprache (*Text2speech*) der Audiodaten der Videos möglich.
