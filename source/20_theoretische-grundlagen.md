\clearpage

# Theoretische Grundlagen \label{chapter_grundlagen}

<!-- is it okay to start withpout this?
Dieser Teil der Arbeit behandelt die notwendigen theoretischen Grundlagen, welche für die im anschließenden Teil behandelten Themen notwendig sind.
 -->

## Künstliche neuronale Netze

<!-- Bernhard TODO: Mehr Literatur? -->
Künstliche neuronale Netze (KNN), auch kurz **neuronale Netze** genannt, bezeichnen einen Ansatz zur Modellierung, welcher im Bereich der künstlichen Intelligenz, genauer im Bereich des maschinellen Lernens seinen Einsatz findet. Das Forschungsgebiet des maschinellen Lernens beschäftigt sich mit einer Klasse von Algorithmen, die anhand von Beispielfällen ein Modell erstellen, welches Inputdaten in Sätze aus Attributen und Eigenschaften kategorisiert. [@News2018]

Ein weiteres Teilgebiet des maschinellen Lernens stellt das tiefe Lernen (engl.: *Deep Learning*) dar. Abbildung \ref{ki_ml_dl} zeigt die Einordnung von neuronalen Netzen und *Deep Learning* in das Gebiet der künstlichen Intelligenz.

![Einordnung neuronaler Netze in die künstliche Intelligenz. *Quelle: Angelehnt an [@News2018]* \label{ki_ml_dl}](source/figures/ki_ml_dl.pdf ){ width=80% }


### Biologisches Vorbild

Der strukturelle Aufbau, sowie die Arbeitsweise von neuronalen Netzen ist der Struktur und Funktionsweise eines Nervensystems, genauer gesagt des menschlichen Gehirns, nachempfunden. Daher wird im Folgenden kurz die Funktionsweise eines biologischen neuronalen Netzwerkes skizziert, wie es zum Beispiel in unserem Gehirn zu finden ist.

"Ein Neuron ist eine Zelle, die elektrische Aktivität sammelt und weiterleitet." [@Kruse2015]

Abbildung \ref{bio_neuron} zeigt ein stark vereinfachtes Modell eines **Neurons**. Hier ist der Zellkörper, der auch Soma genannt wird, zu sehen. Von ihm aus gehen mehrere Dendriten, sowie das Axon ab. Der Zellkörper ist in der Lage eine interne elektrische Spannung zu speichern. Dabei laden elektrische Signale, die über die Dendriten zum Soma transportiert werden, den Zellkörper auf. Ab einem gewissen Schwellwert entlädt sich dieser wieder über das Axon, welches mit Dendriten von anderen Neuronen über die Synapsen verbunden ist und lädt diese dadurch auf. So entsteht ein größeres Netzwerk aus Neuronen.

![Darstellung eines biologischen Neurons. Quelle: [@Kruse2015] \label{bio_neuron}](source/figures/neuron.png){ width=100% }

Die Verbindung zwischen Synapsen und Dendriten ist nicht perfekt leitend, da eine "kleiner Spalt" zwischen ihnen besteht, den die Elektronen nicht ohne Weiteres überwinden können. Dieser ist mit chemischen Substanzen, den sogenannten Neurotransmittern, gefüllt. Diese können durch eine anliegende Spannung ionisiert werden, sodass sie eine Ladung über den Spalt transportieren. [@HeinsohnBoerschSocher2012] [@News2018]

Die Synapsen spielen in diesem neuronalen Netz eine sehr wichtige Rolle. Sie können ihre Leitfähigkeit verändern, wodurch ein neuronales Netz mithilfe der Anpassung der Leitfähigkeit (der Gewichte) der einzelnen Verbindungen (Synapsen) zwischen den Neuronen lernfähig wird. Denn abhängig von der Leitfähigkeit der einzelnen Synapsen, verändert sich die Reaktion des Netzwerkes auf bestimmte Eingabeinformationen.

### Formalisiertes Modell

Aufgrund dieser vereinfacht beschriebenen Funktionsweise lag zunächst die Idee nahe, ein Neuron formal als *Schwellenwertelement* zu modellieren. Bereits 1943 untersuchten McCulloch und Pitts ein solches Modell, weshalb *Schwellenwertelemente* auch *McCullock-Pitts-Neuronen* genannt werden. Oft wird ein *Schwellenwertelement* auch als *Perzeptron* bezeichnet, obwohl dieses von Rosenblatt entworfene Modell eigentlich noch etwas komplexer ist.
Der Aufbau eines solchen künstlichen Neurons wird in Abbildung \ref{perzeptron} gezeigt [@Kruse2015].

![Darstellung eines Perzeptrons - *Angelehnt an [@Kruse2015] * \label{perzeptron}](source/figures/perzeptron.pdf){ width=70% }

<!-- TODO: Adjust Bild und FOrmel Text -->
Den Ausgabewert a eines Neurons erhält berechnet sich durch Anwendung der Aktivierungsfunktion $f_\mathrm{act}$ auf die interne Ladung des Neurons.

 

Bei klassischen *Schwellenwertelementen* ist die Aktivierungsfunktion typischerweise die *Sprungfunktion*, welche durch die folgende Formel beschrieben wird.

$$
 	  f_\mathrm{act}(\epsilon,\theta) = \Bigg\{ \begin{tabular}{ll} $1,$ wenn $\epsilon\geq\theta,$ \\$0,$ sonst. \end{tabular} 
$$ 

Die *Sprungunfktion* nimmt den Wert 1 an, sobald die interne Ladung $\epsilon$ eines Neurons den definierten Schwellenwert $\theta$ überschreitet. 

$$
	\epsilon = \sum_{i=1}^nw_ix_i + b
$$

Die interne Ladung $\epsilon$ wird als die gewichtete Summe der Eingabeparameter berechnet. Also das Skalarprodukt eines Eingabevektors $\vec{x}$ mit den Eingabewerten $x_1, ..., x_n$ und dem Gewichtsvektor $\vec{w}$ mit den jeweiligen Gewichten $w_1, ..., w_n$. Zu diesem wird vor Anwendung der Aktivierungsfunktion noch ein sogenannter Bias $b$ addiert <!-- TODO: describe BIAS?-->.

Der berechnete Ausgabewert $a$ dient entweder als endgültige Ausgabe, oder als ein Eingabewert $x_i$ für eines oder mehrere weitere Neuronen im künstlichen neuronalen Netzwerk.

### Aktivierungsfunktionen

Die Wahl der Aktivierungsfunktion spielt eine wichtige Rolle bei der Modellierung eines KNN´s, denn sie bringt die Eingabewerte in Relation mit dem späteren Ausgabewert des Neurons.
Die Aktivierungsfunktion soll eine nicht-lineare Komponente in das neuronale Netzwerk bringen, da ansonsten ausschließlich lineare Probleme gelöst werden könnten [@Gupta2013].

![Beispiele für Aktivierungsfunktionen. \label{aktfunktionen}](source/figures/aktivierungsfunktionen.png){ width=100% }

Einige Beispiele für im Umfeld von maschinellen Lernen häufig verwendete Aktivierungsfunktionen sind in Abbildung \ref{aktfunktionen} zu sehen. [@Goodfellow-et-al-2016]

### Aufbau und Funktionsweise

Bei der Betrachtung neuronaler Netze werden diese typischerweise als gerichtete Graphen dargestellt. 
Ein Graph besteht im Allgemeinen aus einem oder mehreren Knoten und Kanten. Die Kanten verbinden die einzelnen Knoten. Die Kanten eines Graphen können ungerichtet oder gerichtet sein. Bei einer ungerichteten Kante existiert eine Verbindung zwischen den Knoten in beide Richtungen, bei einem gerichteten jedoch immer nur in eine. Man spricht von einem gerichteten oder ungerichteten Graphen, wenn dieser nur gerichtete oder ungerichtete Kanten enthält. [@Goodfellow-et-al-2016]

Bei der Darstellung eines neuronalen Netzes symbolisieren die Knoten die einzelnen Neuronen und die gerichteten Kanten die Synapsen bzw. die Verbindungen.
Ein neuronales Netz wird in der Regel aufgeteilt in eine Eingabe-, sowie eine Ausgabeschicht (engl.: *Input-/Output-Layer*) und optional eine oder mehrere versteckte Schichten (engl.: *Hidden Layer*). Jede dieser Schichten (engl.: *Layer*) kann ein oder mehrere Neuronen enthalten. Die Anzahl der Schichten wird als die *Tiefe* $L$ des Netzwerkes, wobei die Eingabeschicht nicht berücksichtigt wird.
<!-- Bernhard TODO: Quelle -->

![Neuronales Netz mit drei Eingängen, einer versteckten Schicht und einem Ausgangsknoten - *Angelehnt an [@Ng2017] * \label{simple_nn}](source/figures/simple_nn.pdf){ width=70% }

Abbildung \ref{simple_nn} zeigt ein einfaches neuronales Netz mit drei Eingängen, einem *Hidden Layer* und einem *Output Layer*.

Ein gerne verwendetes Beispiel für neuronale Netze ist die Erkennung von Tieren auf einem Bild.
Anhand des Beispiels eines neuronalen Netzes zur Klassifizierung von Hundebildern soll im Folgenden die grundsätzliche Funktionsweise der einzelnen Schichten beschrieben werden.
Zunächst nimmt die Eingabeschicht die benötigten Informationen von außen entgegen, zum Beispiel die numerisch dargestellten Pixel eines Hundebildes. Die Eingabedaten werden durch die versteckten Schichten geleitet und entsprechend verändert, bis sie zur Ausgabeschicht gelangen, welche nun ein Ergebnis anhand der Eingabewerte liefert. In unserem einfachen Beispiel zur Feststellung, ob sich auf einem Bild ein Hund befindet oder nicht handelt es sich um eine binäre Klassifikation. Das KNN würde eine Zahl zwischen 0 und 1 ausgeben, die der Wahrscheinlichkeit entspricht, dass das eingegebene Bild einen Hund darstellt.

![Hunde-/Katzen-Klassifizierer - *Angelehnt an [@Kirste2018]* \label{hunde_klassifizierer}](source/figures/classifier.pdf){ width=100% }

Bei einer Klassifizierung mit mehr als zwei Klassen (z.B. Hund, Katze oder keines von beidem) entspricht das Ergebnis einem Ausgabevektor aus Wahrscheinlichkeiten für jede Klasse. Die Summe der ausgegebenen Wahrscheinlichkeiten entspricht stets 1. Letzteres Beispiel ist in Abbildung \ref{hunde_klassifizierer} vereinfacht dargestellt.
 
### Training

Eine grundlegende Eigenschaft eines KNN ist,<!--Bernhard TODO:  Quelle ?--> dass es trainiert werden kann. Während der Trainingsphase *lernt* das neuronale Netz anhand von Eingabedaten passende Ausgabedaten zu liefern. 

<!-- NICEMAKING --> 
\clearpage

Das Wort *lernen* ist ein starker Begriff, da leicht die Idee aufkommen könnte, die Maschine (oder das KNN) würde analog zum Menschen eine neue Fertigkeit, wie zum Beispiel Zeichnen oder das Verstehen einer fremden Sprache, erlernen. 
Bei der herkömmlichen Entwicklung von Programmen ist der Großteil des Programmverhaltens durch den Programmierer klar vorgegeben. Das bedeutet, der Entwickler setzt klare Regeln für die Lösung eines entsprechenden Problems. [@Gupta2013]

Beim maschinellen Lernen dagegen werden bestimmte Regeln zur Anpassung von Parametern anhand gegebener Daten verwendet. [@Gupta2013]

Genauer bedeutet das, dass je nach Art der verfügbaren Daten innerhalb eines Trainingsprozesses die einzelnen Gewichte und Biase des neuronalen Netzes anhand von fundierten mathematischen Verfahren angepasst werden. 
Bei den den Trainings- bzw. Lernverfahren wird im Allgemeinen unterschieden zwischen dem überwachten und dem unüberwachten Lernen, sowie dem *Reinforcement Learning*. 


Das überwachte Lernen (engl.: *supervised learning*) ist die einfachste Trainingsmethode<!--Bernhard TODO: warum einfachste--> für neuronale Netze. Hierzu benötigt man einen Datensatz, in welchem sich sowohl die Eingangsdaten, als auch die dazu passende Ausgabe befindet. Hierbei werden in mehreren Durchläufen (Epochen) die Eingabewerte dem neuronalen Netzwerk präsentiert und der *Netzwerkfehler* berechnet. Der Trainingsprozess hat das Ziel den *Verlust* mit jeder Epoche zu verringern und diesen dadurch zu minimieren.
Der *Verlust* kann mit Hilfe von unterschiedlichen Kosten-Funktionen berechnet werden.  <!--Bernhard TODO: Quelle -->
Typische Probleme, welche mit *supervised learning* gelöst werden, sind Probleme der Regression oder Klassifizierung.

Das Gegenstück zum *supervised learning* ist das unüberwachte Lernen (engl.: *unsupervised learning*).
Hierbei werden dem Netzwerk in jedem Schritt Trainingsdaten gezeigt, ohne jedoch den Zielausgabewert zu kennen. Das Netz *lernt* in diesen Daten bestimmte Strukturen oder Muster zu erkennen. 
Beispiele für Probleme des unüberwachten Lernens sind die Erkennung von Ausreißern oder generative Modelle, welche neue Daten nach Art der Trainingsdaten generieren. 
<!-- Bernhard TODO: Bild unsupervised learning -->

Beim **Reinforcement Learning** wird dem neuronalen Netz, ähnlich wie beim überwachten Lernen, während des Lernprozesses ein Feedback gegeben. Die Ausgabe eines Reinforcement Learning Modells wird als *action* bezeichnet. Das Label (Zielwert beim überwachten Lernen) für einen Eingabewert wird *reward* genannt. Das Netz erhält vereinfacht gesagt für jede Eingabe eine Belohnung oder eine Bestrafung. Ein *reward* muss sich nicht immer direkt auf eine Eingabe beziehen, sondern kann sich auch auf mehrere Eingaben, oder eine Eingabe der Vergangenheit beziehen. 

Typische Anwendungsfelder sind zum Beispiel das Spielen eines Spiels (z.B. Go) oder auch die Steuerung von Robotern, bei denen es kein *richtiges* Ergebnis im eigentlichen Sinne gibt, sondern nur Konsequenzen, welche geringfügig mit bestimmten Aktionen in Verbindung stehen [@Gupta2013].

Im weiteren Verlauf wird zunehmend auf überwachtes Lernen eingegangen, da die Methode auch in dieser Arbeit verwendet wird.
Nach der Trainingsphase ist das neuronale Netzwerk im Idealfall in der Lage, anhand von *ungesehenen* Eingaben, das heißt solchen, welche nicht im Trainingsdatensatz vorhanden waren, den richtigen Ausgabewert zu ermitteln. Man nennt das die Generalisierungsfähigkeit des Netzes. Es kann passieren, dass während des Trainings eine Überanpassung (engl.: *overfitting*) an die Daten aus dem Trainingsdatensatz stattgefunden hat. Das bedeutet das Netzwerk kennt die Daten so gut, dass es diese perfekt zuordnen kann, kann jedoch keine brauchbaren Ergebnisse für neue Daten liefern. <!-- Bernhard TODO: welche Daten. Begrifflichkeiten einführen? -->

Daher ist es ein weiteres Ziel der Trainingsphase auch eine Überanpassung zu verhindern und somit eine gute Generalisierungsfähigkeit zu erhalten. [@Kruse2015] <!-- Bernhard TODO: underfitting -->

Nach jeder Epoche des Trainings werden die Gewichte anhand einer sogenannten Lernregel angepasst.
Im Folgenden wird auf einige bekannte Lernregeln kurz eingegangen.

#### Hebb-Regel

Die Hebb-Regel stellt eine der einfachsten Lernregeln dar. Sie weist eine große biologische Plausibilität auf und wurde 1949 vom Psychologen Donald Olding Hebb aufgestellt.
Auf das Thema der neuronalen Netze bezogen, lässt sich die Regel wie folgt formulieren:

*Das Gewicht zwischen zwei Knoten wird dann verändert, wenn beide Knoten gleichzeitig aktiv sind.* [@NeuronalesNetz-de-Hebb]

Als Formel lässt sie sich wie folgt beschreiben:
$$
	\Delta w_{ij} = \alpha a_i a_j
$$

$\Delta w_{ij}$ beschreibt die Größe der Gewichtsanpassung zwischen den beiden Knoten $n_i$ und $n_j$. Die Lernrate $\alpha$ stellt einen sogenannten *Hyperparameter* dar, der bereits vor dem Trainingprozess definiert wird und die gesamte Trainingsphase unverändert bleibt. Sie hat einen direkten Einfluss darauf, wie stark die Gewichte nach jeder Epoche angepasst werden. $a_i$ und $a_j$ stehen für das Aktivitätsniveau des empfangenden und des sendenden Knotens.

#### Delta-Regel

Die Delta Regel, auch *Windroff-Hoff-Regel* genannt, basiert auf dem Vergleich einer zu erwartenden und der tatsächlichen Ausgabe eines Knotens. Folgende Formel beschreibt die Funktionsweise der Delta-Regel.
$$
	\Delta w_{ij} = \alpha \delta_i a_j
$$

<!-- NICEMAKING -->
\clearpage

Die Bedeutung von $\Delta w_{ij}$, $\alpha$, sowie $a_j$ in dieser Formel ist identisch zur *Hebb-Regel*. $\delta_i$ bezeichnet hier die Differenz zwischen dem erwarteten und dem tatsächlichen Aktivitätsniveau des sendenden Knotens $n_i$. Die folgende Formel veranschaulicht die Berechnung von $\delta_i$.

$$
\delta_i = a_i \small (erwartet) - a_i(tats"achlich)
$$
Wie in der Formel zu sehen, ist für die Ermittlung von $\delta_i$, und somit für die Anwendung der *Delta-Regel*, die Kenntnis des zu erwartenden (korrekten) Ausgabewertes des Knotens $n_i$ erforderlich. 

Beim *supervised learning* liegt der korrekte Ausgabewert des gesamten Netzwerks vor. Das bedeutet die *Delta-Regel* ist ausschließlich für einschichtige neuronale Netze, also Netze ohne versteckte Schichten, einsetzbar, da nur hier die Ausgabe des Netzes direkt auf die Ausgabe der einzelnen Knoten zurückzuführen ist.

#### Backpropagation

<!--Jenni TODO: Check after compile (Trennung von Grundidee) -->
Der Backpropagation-Algorithmus hilft dieses Problem zu lösen, sodass die Grund\-idee der Delta-Regel auch auf *tiefe* neuronale Netze, also Netze mit mehreren versteckten Schichten angewendet werden kann. Damit eine Berechnung möglich wird unterteilt der Algorithmus die Anpassung der Gewichte jeweils in die 3 Schritte *forward pass*, *Fehlerermittlung* und *backward pass*. 

1. **Forward pass:** Hier werden dem Netz entsprechende Eingabedaten aus dem Trainings-Datensatz präsentiert und von der Eingabe- bis hin zur Ausgabeschicht die Werte aller Knoten berechnet.

<!-- NICEMAKING -->
\clearpage

2. **Fehlerermittlung:** Hier werden die Fehler der Ausgabeknoten ermittelt. Der Fehlerwert wird nun mit einem definierten Schwellwert verglichen. Ist der Fehler kleiner als der Schwellwert, oder die definierte Anzahl an Epochen bereits erreicht, wird der Algorithmus abgebrochen, falls nicht erfolgt der 3. Schritt.

3. **Backward pass:** Dieser Schritt, stellt die innovative Neuerung des Algorithmus dar. Die zuvor ermittelten Fehler breiten sich jetzt von der Ausgabeschicht, bis hin zur Eingabeschicht rückwärts aus und die Gewichte der einzelnen Knoten werden entsprechend angepasst. Zur Bestimmung der Gewichtsanpassung kommt das Gradientenabstiegsverfahren zum Einsatz. Nach der Anpassung der Gewichte startet der Algorithmus mit der nächsten Trainingsepoche erneut mit dem *Forward pass*.

Auf eine genaue Beschreibung und mathematische Definition des Gradientenabstiegsverfahrens soll aufgrund der Komplexität in dieser Arbeit verzichtet werden. Typischerweise greifen heutige neuronale Netze auf den Backpropagation Algorithmus zurück. [@Rumelhart1986]

### Faltende Neuronale Netze

Bei faltenden neuronalen Netzen (CNN, convolutional neural networks) handelt es sich um eine Sonderform von KNN, welche vor allem bei Daten verwendet werden, die eine Raster-artige Struktur aufweisen, beispielsweise Bilder, welche als ein zweidimensionales Raster von Pixel-Werten beschrieben werden können.

Ein typisches CNN besteht aus einer oder mehreren *convolutional* Schichten gefolgt von einer oder mehreren *fully-connected* Schichten, wie wir sie bereits aus den klassischen neuronalen Netzen kennen.

![Vereinfacht dargestellte Funktion einer *convolutional* Schicht eines faltenden neuronales Netzwerk \label{CNN}](source/figures/cnn.pdf){ width=90% }

Eine *convolutional* Schicht besteht aus einem oder mehreren Filtern gleicher Größe. Man kann sich diesen Filter als eine Art Fenster vorstellen, welches über die Daten "geschoben wird". Dabei entstehen aus den meist größeren Rastern der Eingabedaten neue Raster mit kleineren Dimensionen (siehe Abbildung \ref{CNN}). Gibt es mehrere Filter, werden die entstehenden Ausgabeschichten aufeinander gestapelt. [@Goodfellow-et-al-2016]

Jede dieser *convolutional* Schichten hat mehrere Hyperparamenter, die beeinflussen, welche Dimensionen das nachfolgende Daten-Raster erhält. Diese wären zum Beispiel die *Filtergröße*, sowie die Anzahl der Filter. Daneben ist die *Schrittweite* ein weiterer Parameter, welcher beeinflusst, wie groß die Sprünge sind, in welchen der Filter über die Daten "geschoben" wird. Als letzter Parameter wäre noch ein mögliches *padding* (Füllung) zu nennen. Hierbei wird eine definierte Anzahl an zusätzlichen Zeilen an jeder Seite des Rasters mit Nullen aufgefüllt. Diese Methode dient dazu, dass der Filter auch die äußeren Werte mit einer ähnlichen Gewichtung berücksichtigen kann.

## Emotionserkennung

Bevor man sich mit der automatisierten Erkennung von Emotionen befasst, sollte zunächst die Frage nach der Definition des Begriffs Emotion geklärt werden. Eine einheitliche Definition ist im Bereich der Psychologie sehr umstritten. Im Allgemeinen beschreiben Emotionen jedoch subjektive Empfindungen kürzerer Zeiträume. In der Wissenschaft haben sich vier verschiedene Ansätze zur Entstehung dieser herauskristallisiert. So wird zumeist zwischen dem evolutionstheoretischen, dem stimulativen, dem kognitiven und dem sozial konstruktiven Ansatz unterschieden. [@Schuller2006]

Nachfolgend wird der evolutionstheoretische Ansatz kurz beschrieben, da sich die Arbeit vorwiegend auf diesen bezieht.
Der Ansatz stützt sich auf die Erkenntnisse von Charles Darwin [@CharlesDarwin1872], der die Ansicht vertrat, dass die Emotionen des Menschen ein Ergebnis der Evolution sind. Jede Emotion impliziert ein bestimmtes Verhalten, welches sich auf das Aussterben oder Überleben einer Art auswirkt.

### Einteilung von Emotionen

Um eine Vorhersage bzw. Erkennung der aktuellen Emotion zu bewerkstelligen, ist es nötig diese sinnvoll zu unterscheiden. Für die Einteilung oder Kategorisierung von menschlichen Emotionen gibt es ebenfalls unterschiedliche Ansätze, welche verfolgt werden können.

Im Allgemeinen unterscheidet man zwischen der kategorischen und der dimensionalen Einteilung. Bei Letzterer werden die Emotionen auf einem Spektrum dargestellt. Daher wird niemals eine konkrete Emotion zugeordnet, sondern immer ein Punkt auf der Skala. [@posner_russell_peterson_2005]

Bei der kategorischen Einteilung wird davon ausgegangen, dass es eine endliche Anzahl an wohl definierten menschlichen Emotionen gibt. Diese Einteilung wird insbesondere von Anhängern des evolutionstheoretischen Ansatzes von Emotionen vertreten. Zur Untermauerung des Darwinschen Ansatzes untersuchten einige Forscher unter der Leitung von Dr. Ekman [@Ekman1972] die Gesichtsausdrücke für bestimmte Situationen in einen Eingeborenen-Stamm in Neu Guinea. Dieser hatte zuvor vollkommen isoliert von anderen Gesellschaften gelebt. Somit waren die Reaktionen der Menschen dort nicht auf gesellschaftliche Einflüsse zurückzuführen. Ekman konnte damals bereits anhand des Gesichtsaudrucks vier universelle Emotionen ableiten. Diese waren *Wut*, *Trauer*, *Ekel* und *Fröhlichkeit*.

In weiterführenden Forschungen konnte Ekman die Erkenntnisse vertiefen und entwickelte zusammen mit einigen anderen Wissenschaftlern das *Facial Acting Coding System* (FACS), welche die menschlichen Emotionen in insgesamt sieben Basisemotionen einteilt, welche unabhängig vom gesellschaftlichen Einfluss vorhanden sind. Zusätzlich zu den vier zuvor abgeleiteten Emotionen beinhaltet das FACS noch die Emotionen *Überraschung*, *Verachtung* und *Angst*.

Die Einteilung nach FACS bildet die Basis für die Klassifizierung von Emotionen in dieser Arbeit.

### Gesichtserkennung

Da sich das FACS auf Gesichtszüge bezieht und daher auch die Emotionserkennung in dieser Arbeit anhand von Gesichtsausdrücken stattfindet, ist es sinnvoll das zu erstellende neuronale Netz mit Bildern von Gesichtern als Eingabedaten zu versorgen. Das heißt, dass von einem Bild im Idealfall immer nur der relevante Teil (das Gesicht) ausgeschnitten und dem neuronalen Netz präsentiert wird. Das gilt sowohl für die Trainingsphase, als auch für die spätere Anwendung.

<!-- NICEMAKING -->
\clearpage

Um aus einem größeren Bildausschnitt automatisiert den relevanten Teil, also den Gesichtsausschnitt, zu extrahieren ist es notwendig innerhalb des Bildes das Gesicht und dessen Rahmen (engl. bounding box) zu erkennen.

Die aktuell gängigste Methode zur Gesichtserkennung ist der sogenannte Viola-Jones Detektor [@Shen1997]. Diese Methode gibt nach Eingabe eines größeren Bildes, die genaue Position des Gesichtes in diesem wieder. Dieser Bereich kann dann ausgeschnitten und weiter verwendet werden. Viola und Jones nutzen in ihrem Algorithmus drei unterschiedliche Techniken, die ihn besonders effizient machen. Im Folgenden sollen diese kurz beschrieben werden.

#### Integral Image

Zur weiteren einfachen Verarbeitung der Eingabebilder sollen nur bestimmte Merkmale der Bilder extrahiert werden. Anstelle eines pixelbasierten Ansatzes führten Viola und Jones hierzu den Begriff des *Integralbildes* (engl. integral image) ein. Die dazu extrahierten *Features* ähneln den Haar-Basis Merkmalen, wie sie zuvor auch in anderen Arbeiten verwendet wurden (vergleich [@Papageorgiou]).

Die Wertigkeit eines Pixels im *Integralbild* steht immer im Zusammenhang mit den Pixeln in der unmittelbaren Umgebung. Je höher der Wert ist, desto heller ist der betreffende Bildausschnitt. Auf Basis dieser Informationen haben Viola und Jones drei aufgrund ihrer Ähnlichkeit sogenannte *Haar-like features* definiert, welche ein Gesicht anhand der entsprechenden Helligkeitsunterschiede beschreiben.

![Haar-like features - *two-rectangle*(1,2) & *three-rectangle*(3) & *four-rectangle*(4) - Quelle: [@HaarLike] \label{haar_features}](source/figures/HaarFeatures.pdf){ width=100% }

"Ein *two-rectangle feature* wird beschrieben durch die Differenz aus der Summe der Pixel zwischen zwei rechteckigen Bildausschnitten. Die Ausschnitte haben dieselbe Größe und Form, und Grenzen entweder horizontal oder vertikal aneinander an.
Ein *three-rectangle feature* wird durch die Differenz der Summe aus zwei äußeren Rechtecken und der Summe eines zentrierten Rechteckes berechnet.
Ein *four-rectangle feature* wird letzten Endes durch die Differenz zwischen diagonalen Paaren von Rechtecken beschrieben." [@Shen1997]

Zur Berechnung des *Integralbildes* wird das Ursprungsbild zunächst in ein Graustufen-Bild umgewandelt. Dies ist ein übliches Vorgehen in der automatisierten Bildverarbeitung, da ein solches in der Regel noch alle benötigten Informationen beinhaltet, aber massiv weniger Daten beinhaltet (z.B. 8 Bit pro Pixel statt 24 Bit pro Pixel). Ein Pixel des Integralbildes an der Stelle $(x, y)$ errechnet sich nun aus der Summe aller Pixelwerte eines Rechtecks des Graustufen-Bildes vom Ursprung $(0,0)$ des Bildes bis zum besagten Punkt $(x, y)$. [@Shen1997]

Die Formel zur Berechnung des Pixelwertes an der Stelle $(x, y)$ *Integralbildes* $ii$ aus dem Ursprungs-Graustufenbild $i$ lautet also wie folgt.

$$
ii(x, y) = \sum_{n=0, m=0}^{n\leq x, m\leq y} i(n, m)
$$

![Berechnung eines Integralbildes aus einem Graustufenbild - *Angelehnt an [@Schmidt2014]*\label{integral_image}](source/figures/integral_image.pdf){ width=90% }

In Abbildung \ref{integral_image} ist die Berechnung eines Pixelwertes anhand eines Beispiels zu sehen. Zu erknnen ist, dass $ii(1, 1) = 255 + 222 + 220 + 205 = 902$

Die Vorteile der Nutzung des *Integralbildes* sind die einfache und damit effiziente Berechnungsmethode, sowie die einfache Adaptierbarkeit des Prozesses auf einzelne Bildausschnitte. Des Weiteren ist es möglich, zur Berechnung von Pixelwerten auf zuvor bereits bestimmte Werte zurückzugreifen. Das ist in Abbildung \ref{integral_image_reuse} veranschaulicht. Hier wird verdeutlicht, dass $ii(2, 0) = i(0, 0) + i(1, 0) + i(2, 0) = 255 + 222 + 200$ gleichzusetzen ist mit $ii(2, 0) = ii(1, 0) + i(2, 0) = 277 + 200$.

![Berechnung der nächstfolgenden Pixelsumme des Integralbildes - *Angelehnt an [@Schmidt2014]*\label{integral_image_reuse}](source/figures/integral_image_reuse.pdf){ width=90% }

#### Modifizierter AdaBoost Algorithmus

In seiner ursprünglichen Form dient der *AdaBoost* Algorithmus dazu die Performance eines Klassifzierers im Bereich des maschinellen Lernens zu optimieren (engl. boost). 
Der Netzwerkfehler eines so optimierten Algorithmus (engl. *boosted algirthm*) konvergiert mit der Anzahl der Trainingsdurchläufe exponentiell gegen Null. <!--Bernhard TODO: Quelle -->

In Viola und Jones Arbeit ergaben sich aus jedem 24x24 Pixel Fenster über 180.000 *rectangle-fatures*. Zwar kann jedes einzelne dieser Merkmale sehr effizient berechnet werden, allerdings wäre es ziemlich aufwendig die gesamte Menge an Merkmalen zu berechnen. Sie stellten die Vermutung auf, dass bereits eine geringe Anzahl dieser Merkmale für einen effektiven Klassifizierer genutzt werden können. Viola und Jones nutzen *AdaBoost* daher an zwei Stellen. Zum einen zum Auswählen der zu nutzenden Merkmale und zum anderen zum eigentlichen Trainieren des Klassifizierers. Dazu wurde der *weak classifier* so angepasst, dass er das Rechteck-Merkmal findet, welches die positiven Beispiel-Daten am besten von den negativen Beispiel-Daten separiert [@Shen1997]. In folgender Abbildung \ref{viola_jones_adaboost} ist der modifizierte *AdaBoost* Algorithmus nach Viola und Jones kurz skizziert. 

![Modifizerter AdaBoost Algorithmus nach Viola und Jones. *Quelle: [@Shen1997]* \label{viola_jones_adaboost}](source/figures/viola_jones_adaboost.jpeg){ width=100% }

#### Cascade of classifiers

Die dritte von Viola und Jones eingeführte Technik zur Gesichtserkennung sind die kaskadierenden Klassifizierer. Die Idee hierbei besteht darin mehrere optimierte Klassifizierer für nur ein bestimmtes Merkmal hintereinander zu schalten. Die einzelnen Klassifizierer haben dabei eine Negativ-Erkennungs-Rate (engl. *false positive rate*) gegen 0. So können diese eine große Anzahl an nicht passenden Bildausschnitten aussortieren und nur die passenden an den enstprechenden nächsten Klassifizierer weitergeben. Da diese einfachen Klassifizierungen sehr effizient berechnet werden können, wird verhindert, dass für nicht passende Ausschnitte zu viel Rechenaufwand betrieben wird. Der Begriff Kaskade wurde gewählt, um zu veranschaulichen, dass jeder Klassifizierer nur anhand eines positiven Ergebnisses des vorgeschalteten Klassifizierers weiter arbeitet.

![Kaskadierende Klassifizierer C1 bis Cn - *Angelehnt an [@Shen1997]* \label{cascade_of_classifiers}](source/figures/cascade_of_classifiers.pdf){ width=90% }
