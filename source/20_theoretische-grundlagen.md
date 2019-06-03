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

Hier sieht man den Zellkörper, auch Soma genannt. Von ihm aus gehen mehrere Dendriten, sowie das Axon ab. Der Zellkörper sit in der Lage eine interne elektrische Spannung zu speichern. Dabei laden elektrische Signale, die über die Dendriten zum Soma transportiert werden, diesen auf. Ab einem gewissen Schwellwert jedoch entlädt dieser sich wieder über das Axon, welches mit Dendriten von anderen Neuronen über die Synapsen verbunden ist und läd diese dadurch auf. So entsteht ein größeres Netzwerk aus Neuronen. 
Die Verbindung zwischen Synapsen und Dendriten ist jedoch nicht perfekt leitend, da es einen "kleinen Spalt" zwischen ihnen gibt, welchen die Elektronen nicht ohne Weiteres überwinden können. Dieser ist mit chemischen Substanzen, den sogenannten Neurotransmittern, gefüllt. Diese können durch eine anliegende Spannung ionisiert werden, sodass Sie dann eine Ladung über den Spalt transportieren. [@HeinsohnBoerschSocher2012] [@News2018]

Die Synapsen spielen also in diesem neuronalen Netz eine sehr wichtige Rolle. Sie können Ihre Leitfähigkeit verändern wodurch ein neuronales Netz mithilfe der Anpassung der Leitfähigkeit (der Gewichte) der einzelnen Verbindungen (Synapsen) zwischen den Neuronen lernfähig wird. Denn abhängig von der Leitfähigkeit der einzelnen Synapsen verändert sich die Reaktion des Netzwerkes auf bestimmte Eingabeinformationen.

### Formalisiertes Model

Aufgrund dieser vereinfacht beschriebenen Funktionsweise lag es zunächst nahe ein Neuron, formal als *Schwellenwertelement* zu modellieren. Bereits 1943 untersuchten McCulloch und Pitts ein solches Model, weshalb man *Schwellenwertelemente* auch *McCullock-Pitts-Neuronen* nennt. Oft werden *Schwellenwertelemente* auch als *Perzeptron* bezeichnet, obwohl dieses von Rosenblatt entworfene Model eigentlich noch etwas komplexer ist.
Der Aufbau eines solchen künstlichen Neurons wird in \ref{kuenst_neuron} gezeigt. [@Kruse2015]

![Darstellung eines biologischen Neurons. *Angelehnt anTODO: * \label{bio_neuron}](source/figures/perzeptron.png){ width=100% }

Der Ausgabewert *a* eines Neurons erhält man durch Anwendung der Aktivierungsfunktion $f_\mathrm{act}$ auf die interne Ladung des Neurons. Bei klassischen *Schwellenwertelementen* ist die Aktivierungsfunktion typischerweise die *Sprungfunktion*.

 $$
	  f_\mathrm{act}(\epsilon,\theta) = \Bigg\{ \begin{tabular}{ll} $1,$ wenn $\epsilon\geq\theta,$ \\$0,$ sonst. \end{tabular}
 $$

 Die interne Ladung $\epsilon$ erhält man indem man eine gewichtete Summe der Eingabeparameter berechnet. Also das Skalarprodukt eines Eingabevektors $\vec{x}$ mit den Eingabewerten $x_1, .., x_n$ und dem Gewichtsvektor $\vec{w}$ mit den jeweiligen Gewichten $w_1, ..., w_n$. Zu diesem wird vor Anwendung der Aktivierungsfunktion noch ein sogenannter Bias $b$ addiert.

$$
	\epsilon = \sum_{i=1}^nw_ix_i + b
$$

Der berechnete Ausgabewert *a* dient wiederum als ein Eingabewert *xi* für eines oder mehrere weitere Neuronen im künstlichen neuronalen Netzwerk.

### Aktivierungsfunktionen

Die Wahl der Aktivierungsfunktion spielt eine wichtige Rolle bei der Modellierung eines KNN´s, denn sie bringt die Eingabewerte in Relation mit dem späteren Ausgabewert des Neurons.
Die Aktivierungsfunktion soll eine nicht-lineare Komponente in das neuronale Netzwerk bringen, da es ansonsten ausschließlich möglich wäre linear lösbare Probleme zu lösen. [@Gupta2013]
Einige Beispiele für im Umfeld von machinellen Lernen häufig verwendete Aktivierungsfunktionen sind in \ref{aktfunktionen} zu sehen.

TODO: Abbildung Aktivierungsfunktionen

### Aufbau und Funktionsweise

Bei der Betrachtung neuronaler Netze werden diese typischerweise als gerichtete Graphen dargestellt. 
Ein Graph besteht im Allgemeinen aus einem oder mehreren Knoten und Kanten. Die kanten verbinden die einzelnen Knoten. Die Kanten eines Graphen können können ungerichtet oder gerichtet sein. Bei einer ungerichteten Kante existiert eine Verbindung zwischen den Knoten in beide Richtungen. Man spricht auch von einem gerichteten oder ungerichteten Graphen wenn dieser nur gerichtete oder ungerichtete Kanten enthält. [@Goodfellow-et-al-2016]

Bei der Darstellung eines neuronalen Netzes symbolisieren die Knoten die einzelnen Neuronen und die gerichteten Kanten die Synapsen bzw. die Verbindungen. 
Ein neuronales Netz wird in der Regel aufgeteilt in eine Eingabe-, sowie eine Ausgabeschicht (engl.: *Input-/Output-Layer*) und optional eine oder mehrere versteckte Schichten (engl.: *hidden Layer*). Jede dieser Schichten (engl.: *Layer*) kann ein oder mehrere Neuronen enthalten. Man bezeichnet die Anzahl der Schichten als die *Tiefe*  des Netzwekes mit $L$, wobei die Eingabeschicht nicht berücksichtigt wird.
Abbildung \ref{simple_nn} zeigt ein einfaches neuronales Netz mit 3 Eingängen, einem *hidden Layer* und einem *Output Layer*.

TODO: Abbildung simple_nn

Anhand des Beispiels eines neuronalen Netzes zur Klassifizierung von Hundebildern soll im Folgenden die grundsätzliche Funktionsweise der einzelnen Schichten beschrieben werden.
Zunächst nimmt die Eingabeschicht die benötigten Informationen von au0en entgegen, zum Beispiel die numerische dargestellten Pixel eines Hundebildes. Die Eingabedaten werden durch die versteckten Schichten geleitet und entsprechend verändert bis Sie zur Ausgabeschicht gelangen, welche nun ein Ergebnis anhand der Eingabewerte liefert. In unserem einfachen Beispiel, zur Feststellung ob es sich um ein Hundebild handelt oder nicht (binäre Klassifikation), würde eine Zahl zwischen 0 und 1 ausgegeben, welche der Wahrscheinlichkeit entspricht, dass das eingegebene Bild einen hund darstellt.
Bei einer Klassifizierung mit mehr als 2 Klassen (z.B. Hund, Katze oder keines von beidem) entspricht das Ergebnis einem Ausgabevektor aus Wahrscheinlichkeiten für jede Klasse. Die Summe der ausgegebenen Wahrscheinlichkeiten entspricht stets 1. Letztetes Beispiel ist in  \ref{hunde_klassifizierer} vereinfacht dargestellt. 

TODO: Abbildung CAT/DOG clsassifier (based on) [@Kirste2018 S.30]


### Training

Eine Grundlegende Eigenschaft eines KNN ist, dass man es trainieren kann. Während derTrainingsphase *lernt* das neuronale Netz anhand von Eingabedaten passende Ausgabedaten zu liefern. 

Das Wort *lernen* ist ein starker Begriff, da man leicht auf die Idee kommen könnte, die Maschine (der das KNN) würde analog zum Menschen eine neue Fertigkeit, wie zum Beispiel Zeichnen oder das Verstehen einer fremden Sprache, erlenen. 
Bei der herkömmlichen Entwicklung von Programmen ist der Großteil des Programmverhaltens durch den Programmierer klar vorgegeben. Das bedeutet der Entwickler setzt klare Regeln für die Lösung eines entsprechndes Problems.
Beim machinellen Lernen dagegen verwendet man bestimmte Regeln zur Anpassung von Parametern anhand gegebener Daten. [@Gupta2013]

Genauer bedeutet das, dass je nach Art der verfügbaren Daten innerhalb eines Trainingsprozesses die einzelnen Gewichte und Biase des neuronalen Netzes anhand von bestimmten Regeln angepasst werden. 
Man unterscheidet im Allgemeinen zwischen den folgenden Trainings- bzw. Lernverfahren.

- **Überwachtes Lernen** (*supervised learning*): 

Das überwachte Lernen ist die einfachste Trainingsmethode für neuronale Netze. Hierzu benötigt man einen Datensatz in welchem sich sowohl die Eingangsdaten, sowie auch die dazu passende Ausgabe befindet. Hierbei werden in mehreren Durchläufen (Epochen) die Eingabewerte dem neuronalen Netzwerk präsentiert und der *Netzwerfehler* berechnet. Der Trainingsprozess hat das Ziel den *Netzwerkfehler* mit jeder Epoche zu veringern und diesen dadurch zu minimieren.
Der *Netzwerkfehler* kann mithilfe von unterschiedlichen Kosten-Funktionen berechnet werden. 
Typische Beispiele für Probleme, welche mit *supervised learning* gelöst werden, sind Probleme der Regression oder Klassifizierung.

- **unüberwachtes Lernen** (*unsupervised learning*):
Das Gegenstück zum *supervised learning* ist das *unsupervised learning*.
Hierbei werden dem Netzwerk in jedem Schritt Trainingsdaten gezeigt, ohne jedoch den Zielausgabewert zu kennen. Das Netz *lernt* in diesen Daten bestimmte Strukturen oder Muster zu erkennen. 
Beispiele für Probleme des unüberwachten Lernens sind die Erkennung von Ausreißern oder generative Modelle, welche neue Daten nach Art der Trainingsdaten generieren.

- **Reinforcement learning**:
Beim Reinfocement learning wird dem neuronalen Netz, ähnlich wie beim überwachten Lernen, während des Lernprozesses ein Feedback gegeben. Die Ausgabe eines Reinforcement learning Models wird als *action* bezeichnet. Das Label (Zielwert beim überwachten Lernen) für einen Eingabewert wird als *reward* bezeichnet. Das Netz erhält also vereinfacht gesagt für jede Eingabe eine Belohnung oder eine Bestrafung. Ein *reward* muss sich nicht immer direkt auf eine Eingabe beziehen, sondern kann sich auch auf mehrere Eingaben, oder eine Eingabe der Vergangenheit beziehen.
Typische Anwendungsfelder sind zum Beispiel das spielen eines Spiels (z.B. Go) oder auch die Steuerung von Robotern, bei denen es kein *richtiges* Ergebnis im eigentlichen Sinne gibt, sondern nur Konsequenzen welche geringfügig smit bestimmten Aktionen in Verbindung stehen. [@Gupta2013]

Im weiteren Verlauf wird zunehmend auf überwachtes Lernen eingegangen, da dieses auch in diese Methode auch in dieser Arbeit verwendet wird.
Nach der Trainingsphase ist das neuronale Netzwerk im Idealfall in der Lage anhand von *ungesehenen*, das heißt Eingaben welche nicht im Trainingsdatensatz vorhanden waren, den richtigen AUsgabewert zu ermitteln. Man nennt das die Generalisierungsfähigkeit des Netzes. Es kann passieren, dass während des Trainings eine Überanpassung (engl.: overfitting) an die Daten aus dem Trainingsdatensatz stattgefunden hat. Das bedeutet das Netzwerk kennt die Daten so gut, dass es diese perfekt zuordnen kann, kann jedoch keine brauchbaren Ergebnisse für neue Daten liefern. 
Daher ist es ein weiteres Ziel der Trainingsphase auch ein *overfitting* zu verhindern und somit eine gute Generalisierungsfähigkeit zu erhalten. [@Kruse2015] <!--TODO: ?? -->

Nach jeder Epoche des Trainings werden die Gewichte anhand einer sogenannten Lernregel angepasst.
Im folgenden werde Ich auf einige bekannte Lernregeln kurz eingehen:

#### Hebb Regel

Die Hebb Regel stellt eine der einfachsten Lernregeln dar. Sie weißt eine große biologische Plausibilität auf und wurde 1949 vom Psychologen Donald Olding Hebb aufgestellt. 
Auf das Thema der neuronalen Netze bezogen lässt sich die Regel wie folgt formulieren:

*Das Gewicht zwischen zwei Knoten wird dann verändert, wenn beide Knoten gleichzeitig aktiv sind.* [@NeuronalesNetz-de-Hebb]

Als Formel lässt Sie sich wie folgt beschreiben:
$$
	\Delta w_{ij} = \alpha a_i a_j
$$

$\Delta w_{ij}$ beschreibt die Größe der Gewichtsanpassung zwischen den beiden Knoten $n_i$ und $n_j$. Die Lernrate $\alpha$ stellt einen sogenannten *Hyperparameter* dar, der bereits vor dem Trainingprozess definiert wird und die gesamt Trainingsphase unverändert bleibt. Sie hat einen direkten Einfluss darauf, wie startk die Gewichte nach jeder Epoche angepasst werden. $a_i$ und $a_j$ stehen für das Aktivitätsniveau des empfangenden und des sendenden Knotens.

#### Delta Regel



#### Back propgation


### Deep Learning

### Convolutional Neural Network

### Trainingsmethoden??

### Gesichtserkennung


<!-- expose Ordnung -->
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
