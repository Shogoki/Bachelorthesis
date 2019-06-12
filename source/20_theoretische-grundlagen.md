\clearpage

# Theoretische Grundlagen

Dieser Teil der Arbeit behandelt die notwendigen theoretischen Grundlagen, welche für die im anschließenden Teil behandelten Themen notwendig sind.

## künstliche neuronale Netze

Künstlische neuronale Netze (KNN), auch kurz einfach **neuronale Netze** genannt, bezeichnen einen Ansatz zur Modellierung, welcher im Bereich der künstlichen Intelligenz, genauer im Bereich des maschinellen Lernens seinen Einsatz findet. Das Forschungsgebiet des maschinellen Lernens beschäftigt sich mit einer Klasse von Algorithmen, die anhand von Beispielfällen ein Modell erstellen, welches Inputdaten in Sätze aus Attributen und Eigenschaften kategorisiert[@News2018]. Ein weiteres Teilgebiet unterhalb des maschinellen Lernens stellt das tiefe Lernen, engl.: *Deep Learning*, dar. Abbildung \ref{ki_ml_dl} zeigt die Einordnung von neuronalen Netzen und *Deep Learning* in das Gebiet der künstlichen Intelligenz.

![Einordnung neuronaler Netze in die künstliche Intelligenz. *Quelle: Angelehnt an [@News2018]* \label{ki_ml_dl}](source/figures/ki_ml_dl.pdf){ width=100% }

### Biologisches Vorbild

Der strukturelle Aufbau, sowie die Arbeitsweise von neuronalen Netzen ist der Struktur und Funktionsweise eines Nervensystems, genauer gesagt des menschlichen Gehirns nachempfunden. Daher skizzieren wir im Folgenden kurz die Funktionsweise eines biologischen neuronalen Netzwerkes, wie es zum Beispiel in unserem Gehirn zu finden ist.

"Ein Neuron ist eine Zelle, die elektrische Aktivität sammelt und weiterleitet." [@Kruse2015]
Ein stark vereinfachtes Modell eines **Neurons** ist in Abbildung \ref{bio_neuron} zu sehen.

![Darstellung eines biologischen Neurons. *Quelle: [@Kruse2015]* \label{bio_neuron}](source/figures/neuron.png){ width=100% }

Hier sieht man den Zellkörper, auch Soma genannt. Von ihm aus gehen mehrere Dendriten, sowie das Axon ab. Der Zellkörper ist in der Lage eine interne elektrische Spannung zu speichern. Dabei laden elektrische Signale, die über die Dendriten zum Soma transportiert werden, diesen auf. Ab einem gewissen Schwellwert jedoch entlädt sich dieser wieder über das Axon, welches mit Dendriten von anderen Neuronen über die Synapsen verbunden ist und läd diese dadurch auf. So entsteht ein größeres Netzwerk aus Neuronen. 
Die Verbindung zwischen Synapsen und Dendriten ist jedoch nicht perfekt leitend, da es einen "kleinen Spalt" zwischen ihnen gibt, welchen die Elektronen nicht ohne Weiteres überwinden können. Dieser ist mit chemischen Substanzen, den sogenannten Neurotransmittern, gefüllt. Diese können durch eine anliegende Spannung ionisiert werden, sodass sie dann eine Ladung über den Spalt transportieren. [@HeinsohnBoerschSocher2012] [@News2018]

Die Synapsen spielen also in diesem neuronalen Netz eine sehr wichtige Rolle. Sie können ihre Leitfähigkeit verändern, wodurch ein neuronales Netz mithilfe der Anpassung der Leitfähigkeit (der Gewichte) der einzelnen Verbindungen (Synapsen) zwischen den Neuronen lernfähig wird. Denn abhängig von der Leitfähigkeit der einzelnen Synapsen, verändert sich die Reaktion des Netzwerkes auf bestimmte Eingabeinformationen.

### Formalisiertes Model

Aufgrund dieser vereinfacht beschriebenen Funktionsweise lag es zunächst nahe, ein Neuron formal als *Schwellenwertelement* zu modellieren. Bereits 1943 untersuchten McCulloch und Pitts ein solches Model, weshalb man *Schwellenwertelemente* auch *McCullock-Pitts-Neuronen* nennt. Oft werden *Schwellenwertelemente* auch als *Perzeptron* bezeichnet, obwohl dieses von Rosenblatt entworfene Model eigentlich noch etwas komplexer ist.
Der Aufbau eines solchen künstlichen Neurons wird in Abbildung \ref{perzeptron} gezeigt[@Kruse2015].

![Darstellung eines Perzeptrons - *Angelehnt an [@Kruse2015] * \label{perzeptron}](source/figures/perzeptron.pdf){ width=50% }

Den Ausgabewert *a* eines Neurons erhält man durch Anwendung der Aktivierungsfunktion $f_\mathrm{act}$ auf die interne Ladung des Neurons. Bei klassischen *Schwellenwertelementen* ist die Aktivierungsfunktion typischerweise die *Sprungfunktion*.

 $$
	  f_\mathrm{act}(\epsilon,\theta) = \Bigg\{ \begin{tabular}{ll} $1,$ wenn $\epsilon\geq\theta,$ \\$0,$ sonst. \end{tabular}
 $$

 Die interne Ladung $\epsilon$ erhält man, indem man eine gewichtete Summe der Eingabeparameter berechnet. Also das Skalarprodukt eines Eingabevektors $\vec{x}$ mit den Eingabewerten $x_1, ..., x_n$ und dem Gewichtsvektor $\vec{w}$ mit den jeweiligen Gewichten $w_1, ..., w_n$. Zu diesem wird vor Anwendung der Aktivierungsfunktion noch ein sogenannter Bias $b$ addiert.

$$
	\epsilon = \sum_{i=1}^nw_ix_i + b
$$

Der berechnete Ausgabewert *a* dient wiederum als ein Eingabewert *xi* für eines oder mehrere weitere Neuronen im künstlichen neuronalen Netzwerk.

### Aktivierungsfunktionen

Die Wahl der Aktivierungsfunktion spielt eine wichtige Rolle bei der Modellierung eines KNN´s, denn sie bringt die Eingabewerte in Relation mit dem späteren Ausgabewert des Neurons.
Die Aktivierungsfunktion soll eine nicht-lineare Komponente in das neuronale Netzwerk bringen, da es ansonsten ausschließlich möglich wäre, linear lösbare Probleme zu lösen[@Gupta2013].
Einige Beispiele für im Umfeld von machinellen Lernen häufig verwendete Aktivierungsfunktionen sind in Abbildung \ref{aktfunktionen} zu sehen.

![Beispiele für Aktivierungsfunktionen.\label{aktfunktionen}](source/figures/aktivierungsfunktionen.png){ width=100% }

### Aufbau und Funktionsweise

Bei der Betrachtung neuronaler Netze werden diese typischerweise als gerichtete Graphen dargestellt. 
Ein Graph besteht im Allgemeinen aus einem oder mehreren Knoten und Kanten. Die Kanten verbinden die einzelnen Knoten. Die Kanten eines Graphen können ungerichtet oder gerichtet sein. Bei einer ungerichteten Kante existiert eine Verbindung zwischen den Knoten in beide Richtungen. Man spricht auch von einem gerichteten oder ungerichteten Graphen, wenn dieser nur gerichtete oder ungerichtete Kanten enthält [@Goodfellow-et-al-2016].

Bei der Darstellung eines neuronalen Netzes symbolisieren die Knoten die einzelnen Neuronen und die gerichteten Kanten die Synapsen bzw. die Verbindungen. 
Ein neuronales Netz wird in der Regel aufgeteilt in eine Eingabe-, sowie eine Ausgabeschicht (engl.: *Input-/Output-Layer*) und optional eine oder mehrere versteckte Schichten (engl.: *hidden Layer*). Jede dieser Schichten (engl.: *Layer*) kann ein oder mehrere Neuronen enthalten. Man bezeichnet die Anzahl der Schichten als die *Tiefe* des Netzwerkes mit $L$, wobei die Eingabeschicht nicht berücksichtigt wird.
Abbildung \ref{simple_nn} zeigt ein einfaches neuronales Netz mit 3 Eingängen, einem *hidden Layer* und einem *Output Layer*.

![neuronales Netz mit 3 Eingängen, 1 versteckten Schicht und einem Ausgangsknoten - *Angelehnt an [@Ng2017] * \label{simple_nn}](source/figures/simple_nn.pdf){ width=70% }

Anhand des Beispiels eines neuronalen Netzes zur Klassifizierung von Hundebildern soll im Folgenden die grundsätzliche Funktionsweise der einzelnen Schichten beschrieben werden.
Zunächst nimmt die Eingabeschicht die benötigten Informationen von außen entgegen, zum Beispiel die numerisch dargestellten Pixel eines Hundebildes. Die Eingabedaten werden durch die versteckten Schichten geleitet und entsprechend verändert, bis sie zur Ausgabeschicht gelangen, welche nun ein Ergebnis anhand der Eingabewerte liefert. In unserem einfachen Beispiel zur Feststellung, ob es sich um ein Hundebild handelt oder nicht (binäre Klassifikation), würde eine Zahl zwischen 0 und 1 ausgegeben, welche der Wahrscheinlichkeit entspricht, dass das eingegebene Bild einen Hund darstellt.
Bei einer Klassifizierung mit mehr als 2 Klassen (z.B. Hund, Katze oder keines von beidem) entspricht das Ergebnis einem Ausgabevektor aus Wahrscheinlichkeiten für jede Klasse. Die Summe der ausgegebenen Wahrscheinlichkeiten entspricht stets 1. Letzteres Beispiel ist in \ref{hunde_klassifizierer} vereinfacht dargestellt. 

TODO: Abbildung CAT/DOG clsassifier (based on) [@Kirste2018 S.30]


### Training

Eine grundlegende Eigenschaft eines KNN ist, dass man es trainieren kann. Während der Trainingsphase *lernt* das neuronale Netz anhand von Eingabedaten passende Ausgabedaten zu liefern. 

Das Wort *lernen* ist ein starker Begriff, da man leicht auf die Idee kommen könnte, die Maschine (oder das KNN) würde analog zum Menschen eine neue Fertigkeit, wie zum Beispiel Zeichnen oder das Verstehen einer fremden Sprache, erlernen. 
Bei der herkömmlichen Entwicklung von Programmen ist der Großteil des Programmverhaltens durch den Programmierer klar vorgegeben. Das bedeutet, der Entwickler setzt klare Regeln für die Lösung eines entsprechenden Problems.
Beim machinellen Lernen dagegen verwendet man bestimmte Regeln zur Anpassung von Parametern anhand gegebener Daten[@Gupta2013].

Genauer bedeutet das, dass je nach Art der verfügbaren Daten innerhalb eines Trainingsprozesses die einzelnen Gewichte und Biase des neuronalen Netzes anhand von bestimmten Regeln angepasst werden. 
Man unterscheidet im Allgemeinen zwischen den folgenden Trainings- bzw. Lernverfahren.

- **Überwachtes Lernen** (*supervised learning*): 

Das überwachte Lernen ist die einfachste Trainingsmethode für neuronale Netze. Hierzu benötigt man einen Datensatz, in welchem sich sowohl die Eingangsdaten, sowie auch die dazu passende Ausgabe befindet. Hierbei werden in mehreren Durchläufen (Epochen) die Eingabewerte dem neuronalen Netzwerk präsentiert und der *Netzwerkfehler* berechnet. Der Trainingsprozess hat das Ziel den *Netzwerkfehler* mit jeder Epoche zu verringern und diesen dadurch zu minimieren.
Der *Netzwerkfehler* kann mit Hilfe von unterschiedlichen Kosten-Funktionen berechnet werden. 
Typische Beispiele für Probleme, welche mit *supervised learning* gelöst werden, sind Probleme der Regression oder Klassifizierung.

- **unüberwachtes Lernen** (*unsupervised learning*):
Das Gegenstück zum *supervised learning* ist das *unsupervised learning*.
Hierbei werden dem Netzwerk in jedem Schritt Trainingsdaten gezeigt, ohne jedoch den Zielausgabewert zu kennen. Das Netz *lernt* in diesen Daten bestimmte Strukturen oder Muster zu erkennen. 
Beispiele für Probleme des unüberwachten Lernens sind die Erkennung von Ausreißern oder generative Modelle, welche neue Daten nach Art der Trainingsdaten generieren.

- **Reinforcement learning**:
Beim *reinforcement learning* wird dem neuronalen Netz, ähnlich wie beim überwachten Lernen, während des Lernprozesses ein Feedback gegeben. Die Ausgabe eines Reinforcement learning Models wird als *action* bezeichnet. Das Label (Zielwert beim überwachten Lernen) für einen Eingabewert wird als *reward* bezeichnet. Das Netz erhält also vereinfacht gesagt für jede Eingabe eine Belohnung oder eine Bestrafung. Ein *reward* muss sich nicht immer direkt auf eine Eingabe beziehen, sondern kann sich auch auf mehrere Eingaben, oder eine Eingabe der Vergangenheit beziehen.
Typische Anwendungsfelder sind zum Beispiel das Spielen eines Spiels (z.B. Go) oder auch die Steuerung von Robotern, bei denen es kein *richtiges* Ergebnis im eigentlichen Sinne gibt, sondern nur Konsequenzen welche geringfügig mit bestimmten Aktionen in Verbindung stehen[@Gupta2013].

Im weiteren Verlauf wird zunehmend auf überwachtes Lernen eingegangen, da die Methode auch in dieser Arbeit verwendet wird.
Nach der Trainingsphase ist das neuronale Netzwerk im Idealfall in der Lage, anhand von *ungesehenen* Eingaben, das heißt solchen, welche nicht im Trainingsdatensatz vorhanden waren, den richtigen AUsgabewert zu ermitteln. Man nennt das die Generalisierungsfähigkeit des Netzes. Es kann passieren, dass während des Trainings eine Überanpassung (engl.: overfitting) an die Daten aus dem Trainingsdatensatz stattgefunden hat. Das bedeutet das Netzwerk kennt die Daten so gut, dass es diese perfekt zuordnen kann, kann jedoch keine brauchbaren Ergebnisse für neue Daten liefern. 
Daher ist es ein weiteres Ziel der Trainingsphase auch ein *overfitting* zu verhindern und somit eine gute Generalisierungsfähigkeit zu erhalten[@Kruse2015]. <!--TODO: ?? -->

Nach jeder Epoche des Trainings werden die Gewichte anhand einer sogenannten Lernregel angepasst.
Im folgenden werde ich auf einige bekannte Lernregeln kurz eingehen:

#### Hebb-Regel

Die Hebb-Regel stellt eine der einfachsten Lernregeln dar. Sie weißt eine große biologische Plausibilität auf und wurde 1949 vom Psychologen Donald Olding Hebb aufgestellt. 
Auf das Thema der neuronalen Netze bezogen, lässt sich die Regel wie folgt formulieren:

*Das Gewicht zwischen zwei Knoten wird dann verändert, wenn beide Knoten gleichzeitig aktiv sind*[@NeuronalesNetz-de-Hebb]. 

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

Die Bedeutung von $\Delta w_{ij}$, $\alpha$, sowie $a_j$ in dieser Formel ist identisch zur *Hebb-Regel*. $\delta_i$ bezeichnet hier die Differenz zwischen dem erwarteten und dem tatsächlichen Aktivitätsniveau des sendenden Knotens $n_i$. Die folgende Formel veranschaulicht die Berechnung von $\delta_i$.
$$
\delta_i = a_i \small (erwartet) - a_i(tats"achlich)
$$

Wie man sieht, ist für die Ermittlung von $\delta_i$, und somit für die Anwendung der *Delta-Regel*, die Kenntnis des zu erwartenden (korrekten) Ausgabewertes des Knotens $n_i$ erforderlich. 
Beim *supervised learning* liegt der korrekte Ausgabewert des gesamten Netzwerks vor. Das bedeutet die *Delta-Regel* ist ausschließlich für einschichtige neuronale Netze, also Netze ohne versteckte Schichten, einsetzbar, da nur hier die Ausgabe des Netzes direkt auf die Ausgabe der einzelnen Knoten zurückzuführen ist.

#### Backpropgation

Der Backpropagation Algorithmus soll dieses Problem lösen, sodass man die Grundidee der Delta-Regel auch auf *tiefe* neuronale Netze, also Netze mit mehreren versteckten Schichten anwenden kann. Damit eine Berechnung möglich wird unterteilt der Algorithmus die Anpassung der Gewichte in jeweils 3 Schritte:

1. **Forward pass** Hier werden dem Netz entsprechende Eingabedaten aus dem Trainings-Datensatz präsentiert und von der Eingabe- bis hin zur Ausgabeschicht die Werte aller Knoten berechnet.

2. **Fehlerermittlung** Hier werden die Fehler der Ausgabeknoten ermittelt. Der Fehlerwert wird nun mit einem definierten Schwellwert verglichen. Ist der Fehler kleiner als der Schwellwert, oder die definierte Anzahl an Epochen bereits erreicht, wird der Algorithmus abgebrochen, falls nicht erfolgt der 3. Schritt.

3. **Backward pass** Dieser Schritt, stellt die innovative Neuerung des Algorithmus dar. Die zuvor ermittelten Fehler breiten sich jetzt von der Ausgabeschicht, bis hin zur Eingabeschicht rückwärts aus und die Gewichte der einzelnen Knoten werden entsprechend angepasst. Zur Bestimmung der Gewichtsanpassung kommt das Gradientenabstiegsverfahren zum Einsatz. Nach der Anpassung der Gewichte startet der Algorithmus mit der nächsten Trainingsepoche erneut mit dem *Forward pass*.

Auf eine genaue Beschreibung und mathematische Definition des Gradientenabstiegsverfahrens soll aufgrund der Komplexität in dieser Arbeit verzichtet werden. 
Typischerweise greifen Heutige neuronale Netze auf den Backpropagation Algorithmus zurück[@NeuronalesNetz-backProp]

### Faltende Neuronale Netze 

Bei faltenden neuronale Netzen (CNN, convolutional neural networks) handelt es sich um eine Sonderform von KNN, welche vor allem bei Daten verwendet werden, welche eine Raster-artige Struktur aufweisen. Beispiele dafür sind zum Beispiel Bilder, welche man sich als ein zweidimensionales Raster von Pixel-Werten vorstellen kann.
Ein typisches CNN besteht aus einer oder mehreren *convolutional* Schichten gefolgt von einer oder mehreren *fully-connected* Schichten, wie wir Sie bereits aus den klassischen neuronalen Netzen kennen. 
Eine *convolutional* Schicht besteht aus einem oder mehreren Filtern gleicher Größe. Man kann diesen Filter als eine Art Fenster vorstellen, welches über die Daten "geschoben werden". Dabei entstehen aus den meist größeren Rastern der Eingabedaten, neue Raster mit kleineren Dimensionen (siehe \ref{CNN}). Gibt es mehrere Filter, werden die entstehenden Ausgabeschichten aufeinander gestapelt[@Goodfellow-et-al-2016].

TODO: Abbildung CNN

Jede dieser *convolutional* Schichten hat mehrere (Hyper-)Paramenter, welche Beeinflussen welche Dimensionen das nachfolgende Daten-Raster erhält. Diese wären zum Beispiel die *Filtergröße*, sowie die Anzahl der Filter. Daneben ist die *Schrittweite* ein weitere Parameter, welche beeinflusst, wie groß die Sprünge sind, in welchen der Filter über die Daten "geschoben" wird. Als letzter Parameter wäre noch ein mögliches *padding* (Füllung) zu nennen. Hierbei wird eine definierte Anzahl an zusätzlichen Zeilen an jeder Seite des Rasters mit Nullen aufgefüllt. Diese Methode dient dazu, dass der Filter auch die äußeren Werte mit einer Ähnlichen Gewichtung berücksichtigen kann.

## Emotionserkennung

Bevor man sich mit der automatisierten Erkennung von Emotionen befassen sollte zunächst die Frage nach der Definition des Begriffs Emotion geklärt werden. Eine einheitliche Definition ist im Bereich der Psychologie sehr umstritten. Im Allgemeinen beschreiben Emotionen jedoch subjektive Empfindungen kürzerer Zeiträume. In der Wissenschaft haben sich vier verschiedene Ansätze zur Entstehung dieser herauskristallisiert. So unterscheidet man zumeist zwischen dem evolutionstheoretischem , dem stimulativen, dem kognitiven und dem sozial konstruktiven Ansatz[@Schuller2006].

Folgend wird der evolutionstheoretische Ansatz kurz beschrieben, da diese Arbeit sich vorwiegend auf diesen stützt.
Dieser stützt sich auf die Erkenntnisse von Charles Darwin [@CharlesDarwin1872], der die Ansicht vertrat, dass die Emotionen des Menschen ein Ergebnis der Evolution sind. Jede Emotion impliziert ein bestimmtes Verhalten, welches sich auf das Aussterben oder Überleben einer Art auswirkt.

### Einteilung von Emotionen

Um eine Vorhersage, bzw. Erkennung der aktuellen Emotion zu bewerkstelligen ist es nötig diese sinnvoll zu Unterscheiden. Für die Einteilung oder Kategorisierung von menschlichen Emotionen gibt es ebenfalls unterschiedliche Ansätze welche verfolgt werden können.
Im Allgemeinen unterscheidet man zwischen der kategorischen und der dimensionalen Einteilung. Bei letzterer werden die Emotionen auf einem Spektrum dargestellt. Es wird also niemals eine konkrete Emotion zugeordnet, sondern immer ein Punkt auf der Skala. 
<!-- TODO: circumplex Skala + cite evtl. Abbildung?-->

Bei der kategorischen Einteilung geht man davon aus, dass es eine endliche Anzahl an wohl definierten menschlichen Emotionen gibt. Insbesondere Vertreter des evolutionstheoretischen Ansatzes von Emotionen gehen auch von einer kategorischen Einteilung aus. Zur Untermauerung des Darwinschen Ansatzes untersuchten einige Forscher unter der Leitung von Dr. Ekman [@Ekman1972] die Gesichtsausdrücke für bestimmte Situationen in einen Eingeborenen-Stamm in Neu Guinea Dieser hatte zuvor vollkommen isoliert von anderen Gesellschaften gelebt. Somit waren die Reaktionen der Menschen dort nicht auf gesellschaftliche Einflüsse zurück zu führen. Ekman konnte damals bereits anhand des Gesichtsaudrucks vier universelle Emotionen ableiten. Diese waren *Wut*, *Trauer*, *Ekel* und *Fröhlichkeit*. 
In weiterführenden Forschungen konnte Ekman die Erkenntnisse vertiefen und entwickelte zusammen mit einigen anderen Wissenschaftlern das *Facial Acting Coding System* (FACS) welche die menschlischen Emotionen in insgesamt sieben Basisemotionen einteilt, welche unabhängig vom gesellschaftlichen Einfluss vorhanden sind. Zusätzlich zu den vier zuvor abgeleiteten Emotionen beinhaltet das FACS noch die Emotionen *Überaschung*, *Verachtung* und *Angst*.
Das FACS bildet die Basis für die Klassifizerung von Emotionen in dieser Arbeit. <!-- TODO: FACS cite. +  evtl Abbildung -->


### Gesichtserkennung

TBD

#### Haar-Cascade 

TBD



TBD: Wie  teilt man Emotionen ein? --> Ekman.

<!-- expose Ordnung 
## Einordnung der Daten

### Beschaffenheit der Daten

### Vorbereitung der Daten

### Klassifizierung

#### Einteilung der Emotionen

#### Zuordnung der Daten

## künstliche neuronale Netze

### Übersicht & Definition

### Deep Learning

### Convolutional Neural Network

### Trainingsmethoden??

### Gesichtserkennung

## aktuell bewährte Modelle
-->