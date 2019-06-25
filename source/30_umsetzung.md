\clearpage

# Umsetzung & Evaluierung

In diesem Teil wird die Umsetzung der Arbeit eingegangen. im ersten Abschnitt wird auf die Art und Einteilung der zu klassifizierenden Daten eingegangen. Im Weiteren Verlauf wird auf die Beschaffung Der verschiedenen Datensätze eingegangen mit welchen gearbeitet werden soll. Anschließend wird auf diverse Verfahren der Datenpräparation wie Vorbereitung, Normalisierung und Datenvervielfältigung eingegangen. Letzten Endes werden verschiedene neuronale Netze entworfen, trainiert und getestet, bis dann im letzten Abschnitt auf die Realisierung des zu erstellenden Webservice eingegangen wird.

## Beschaffenheit und Einteilung der Daten

Wie bereits beschrieben sollen für den späteren Webservice Videodaten als Eingabe dienen. Das neuronale Netz wird jedoch nicht direkt die Videodaten, sondern aus dem Video extrahierte Einzelbilder als Eingabe erhalten. Diese Eingabedaten werden anhand später beschriebener Verfahren noch weiter vorbereitet und optimiert.
Die Einzelbilder sollen als *Array* von Pixelwerten eines Graustufenbildes an das neuronale Netzwerk übergeben werden (siehe Datenpräparation).
Die Bilder sollen in 7 verschiedene Klassen. Die verschiedenen Emotionen bilden den Zielklassenvektor $Z$, es gilt also $|Z| = 7$. Jedes Bild welches der selben Emotion des *FACS* entspricht ist Mitglied der selben Klasse aus $Z$.

## Datensätze

Ein Teil der Arbeit bestand darin geeignete Datensätze für das Training und die Evaluierung des neuronalen Netzes zu finden und diese später in entsprechende Trainings, Evaluierungs und Test-Datensätze für das neuronale Netzwerk zu unterteilen.

### Beschaffung der Datensätze

Dazu sollen 2 verschiedene Ansätze unterschieden werden. Zum einen die Beschaffung eines vorhandenen freien Datensatzes von Gesichtsbildern inklusive der Zuordnung zu einer der entsprechenden *FACS* Emotionen, sowie zum anderen die Generierung von eignen Daten mithilfe eines Webservice und freiwillgen Probanden. 
In dieser Arbeit wurden beide Ansätze in Kombination verwendet. Es wurden also 2 verschiedene Datenquellen verwendet, was bei der Aufteilung der Datensätze noch eine wichtige Rolle spielt (siehe Einteilung der Datensätze<!--TODO: this may changes -->). 

#### FER+

Als großer frei Verfügbarer Datensatz wurde der *Facial Expression Recognition+* (FER+)[@Barsoum2016] Datensatz verwendet. Bei den Eingangsaten des *FER+* handelt es sich um die selben Bilder, wie auch beim *FER2013*, welcher Teil der International Conference for Machine Learning (ICML) Challenge 2013 war, und danach der Öffentlichkeit zur Verfügung gestellt wurde. Bei FER+ wurden jedoch alle *Label*, mithilfe von *Crowdsourcing* neu erstellt, um eine bessere Datenqualität zu erreichen. (Vergleich [@Barsoum2016]). Der Datensatz besteht aus 34034 48x48 Graustufen Bilder von Gesichtern. Jedes dieser Bilder von je 10 Taggern mithilfe von *Crowdsourcing* bewertet. Der Datensatz enthält für jede Klasse (Emotionen des *FACS* und "kein Gesicht") die Anzahl an Taggern, welche das Bild entsprechend bewertet haben.
Ein Beispiel für ein einzelnes Datum des Datensatzes ist in Abbildung \ref{single_ferplus} zu sehen

TODO: Abbildung FER+ Single row image

Das Team von Microsoft Research[@Barsoum2016] beschreibt mehrere Variationen, wie die mehrfach *gelabelten* Daten verwendbar sind. In dieser Arbeit wird jedoch ausschließlich der einfache Mehrheits-Ansatz verfolgt. Es wird also jedes Bild der Klasse zugeordnet, welche die meisten Stimmen erhalten hat.

#### selbsterstellte Daten

Zum selbst erstellen von Daten wurde im Rahmen der Arbeit eine einfache Website erstellt, welche mithilfe von *WebRTC* Zugriff auf die Kamera bekommt. Auf dieser Website haben freiwillige Probanden, und auch der Autor dieser Arbeit, die Möglichkeit nacheinander für jede Emotion des *FACS* ein 15 sekündiges Video aufzunehmen. Dieses wird anschließend direkt auf dem Server gespeichert. Die Webseite ist in Abbildung \ref{webrtc_screenshot} zu sehen.

![Screenshot der Video-Recording Website\label{bio_neuron}](source/figures/web_recorder.png){ width=70% }

Mithilfe dieser Webseite wurden insgesamt 25 Sätze an 15 sekündigen Videos von 10 verschiedenen freiwilligen Probanden (den Autor dieser Arbeit eingeschlossen) gesammelt. Aus diesen Videos wurde anschließend mit Hilfe des folgenden Python-Skripts pro Sekunde ein Einzelbild extrahiert, und mit dem Namen der entsprechenden Klasse abgespeichert. Somit wurden also $25 * 15 = 375$ Einzelbilder pro Klasse generiert. 

```python

def extract_video_frames(prefix, videofile,
	targetdir = "../data/extracted"):

    vidcap = cv2.VideoCapture(videofile)
	fps = min((50, int(vidcap.get((cv2.CAP_PROP_FPS)))))
	print ("extracting every {} frame from {}".format(fps,videofile))
    success,image = vidcap.read()
    count = 0
    while success:
        if (count % fps) == 0:
			# save frame as JPEG file
            cv2.imwrite("{}/{}_frame{}.jpg".format(targetdir,
				prefix , count) , image)  
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
```
Python-Skript zum extrahieren der Einzelbilder\label{listing_extract_video}

Die Einzelbilder wurden anschließend vom Autor manuell auf Korrektheit, das heißt Zuordnung zur Klasse, geprüft. Dabei wurden insgesamt 100 Bilder wieder aussortiert (TODO: Verify numbers).


### Einteilung der Datensätze

Beim machinellen Lernen ist es üblich den vorhandenen Datensatz, bzw. die vorhandenen Datensätze in verschiedene Verwendungszwecke einzuteilen. Klassischerweise sprach man hier immer vom *train/test-Split*, also einer Aufteilung der Daten in einen Trainings- und einen Test-Datensatz. In modernen Projekten, welche sich mit maschinellen Lernen beschäftigen spricht man jedoch zumeist von einem *train/dev/test-split*. Die Daten werden also in einen Trainings-, einen Entwicklungs- und einen Test-Datensatz eingeteilt. Als Entwicklungs-Datensatz bezeichnet man, jene Daten, welche während der Entwicklung, also dem Anpassen bestimmter (Hyper-)Parameter, des neuronalen Netzes zur Evaluierung verwendet. Der Test-Datensatz ist in diesem Szenario ein Satz aus Daten, welches das neuronale Netz vor der Fertigstellung noch nicht "zu sehen" bekommen hat. Beim klassischen *train/test-split* ist der Test-Satz also eigentlich, das was wir Heute als Entwicklungs-Datensatz bezeichnen, und es gibt keinen wirklichen Test-Datensatz.
Bei der Wahl der Datenquellen, ist es wichtig, dass die Test-Daten möglichst ähnlich, zu den später erwarteten Eingangsdaten sind, und dass Entwicklungs- und Test-Datensatz aus der selben Quelle stammen sollten.

Für diese Arbeit bedeutet das, dass die Entwicklungs- und Test-Daten aus den selbstgenerierten Daten stammen, da diese Bereits von aufgenommenen Videos stammen, was den Zeildaten sehr nahe kommt.
Als Trainingsdaten wird entsprechend der *FER+* Datensatz verwendet.
Ein Problem bei einer solchen Aufteilung, wenn also die Trainingsdaten aus einem anderen Datensatz stammen als die Entwicklungs und Test-Daten ist, dass man gewisse Probleme, wie zum Beispiel eine Überanpassung manchmal nur schwer erkennen kann. Deshalb ist es in einem solchen Fall sinnvoll noch einen vierten Datensatz einzuführen, welcher aus der selben Quelle wie die Trainingsdaten stammt (hier *FER+*). Man spricht hier vom *dev_train* oder auch *bridge* Datensatz. Dieser wird im Prinzip analog zum Entwicklungsdatensatz behandelt, und dient zum testen der Parameter des neuronalen Netzwerkes, nach jeder Änderung. Anhand der Unterschiedlichen Ergebnisse für den *bridge* und den *dev* Datensatz kann man nun schnell, bestimmte Probleme des neuronalen Netzwerks erkennen.

In dieser Arbeit wurde daher auch die Einteilung in 4 Datensätze gewählt. Die Daten wurden dabei wie folgt aufgeteilt:
Die selbsterstellten Daten wurden in zu je 50% in den Entwicklungs- und Test-Datensatz aufgeteilt. Vom *FER+* Datensatz wurden  10% Der Bilder für den *Bridge* Datensatz verwendet und 90% als Trainingsdaten. Die Aufteilung ist in Abbildung \ref{data_split} veranschaulicht.

![Aufteilung der Daten in 4 Datensätze \label{data_split}](source/figures/train_test_split.pdf){ width=90% } <!-- TODO: enter number -->
 <!--like this https://www.freecodecamp.org/news/what-to-do-when-your-training-and-testing-data-come-from-different-distributions-d89674c6ecd8/ -->

Um eine Eingangs zufällige, jedoch immer gleich reproduzierbare Aufteilung zu erzielen wurde das folgende Python Skript verwendet.

TODO: Code Listing Data_split.py


## Datenpräparation

Um die Effizienz, sowie die Genauigkeit der Vorhersage des neuronalen Netzwerkes zu steigern werden alle Daten, bevor Sie dem KNN präsentiert werden auf diverse Arten präpariert. Im Folgenden wird auf die angewandten Methoden genauer eingegangen.

### Vorverarbeitung

Um ein möglichst gutes Ergebnis zu erzielen wurden die selbst erstellten Daten, aber auch die frei verfügbaren Daten aus dem *FER+* Datensatz auf diverse Weise Vorverarbeitet. 
Bei den selbst aufgezeichneten Daten, wurde wie bereits erwähnt eine abschließende manuelle Sichtung vorgenommen, um möglichst alle falsch markierten Daten auszusortieren. Des weiteren werden alle der selbst aufgezeichneten Bilder auf den Ausschnitt des Gesichtes beschränkt, bevor Sie dem selbst entworfenen KNN präsentiert werden. Dazu wird der bereits beschriebene *Viola-Jones-Detektor*[@Shen1997] verwendet um das Gesicht im jeweiligen Bild zu detektieren. Nachdem der gesichtsenthaltende Teil des Bildes ausgeschnitten wurde, wird dieser auf 48x48 Pixel skaliert und in ein Graustufenbild umgewandelt. Damit haben letzten Endes alle Eingangsdaten des neuronalen Netzes die gleiche Struktur. Der Python Code für diese Vorverarbeitung ist im folgenden Listing zu sehen.

TODO: Listing, preprocess selfrecorded data.

Bei den Daten aus dem *FER+* Datensatz ist etwas weniger Vorverarbeitung nötig.Die Vorverarbeitung dieser Daten besteht im wesentlcihen darin, die Pixelwerte welche als ein eindimensionales Array vorliegen in eine 48x48 Matrix umzuwandeln. Des weiteren wird aus dem mehrstimmigen *Labels* des *FER+* Datensatzes noch mithilfe des einfachen Mehrheitsprinzips das hier genutzte extrahiert. In einem letzten Schritt werden dann noch alle Datensätze, welche mit "not a face" markiert sind, aussortiert. (siehe Listing TODO:) 

TODO: CODELISTING PROCESS FER+


### Normalisierung

*"Data normalization has been proposed to address the aforementioned challenge by reducing the training space and making the
training more efficient."* [@Zhang2018]

Ein üblicher Schritt, um die Trainingsphase  im machinellen Lernen zu beschleunigen ist es die Eingabedaten zu normalisieren. Ziel ist es die Eingabedaten, welche auf einem sehr breiten Spektrum liegen so zu normalisieren um das Spektrum zu verkleinen. In unserem Fall geht es um die Graustufenbilder. Im Generellen kann man Normalisierung von solchen Bilden wie folgt beschreiben: Ein n-dimensionales Graustufenbild $I:\{\mathbb{X}\subseteq\mathbb{R}^n\}\rightarrow\{\text{Min},..,\text{Max}\}$ mit den Pixelwerten zwischen $Min$ unx $Max$ wird in ein neues Graustufenbild $I_N:\{\mathbb{X}\subseteq\mathbb{R}^n\}\rightarrow\{\text{newMin},..,\text{newMax}\}$ mit Pixelwerten zwischen $newMin$ und $newMax$ überführt.[@gonzalez2008digital]
Die lineare Noramlisierung eines Graustufenbildes berechnet sich wie folgt:

$$
I_N=(I-\text{Min})\frac{\text{newMax}-\text{newMin}}{\text{Max}-\text{Min}}+\text{newMin}
$$

In unserem Beispiel sind die Ausgangswerte für $Min = 0$ und $Max = 255$ und für eine einfache Normalisierung wählen wir die Werte $newMin = 0$ und $newMax = 1$, damit alle Werte zwischen 0 und 1 liegen. Damit ergibt sich die vereinfachte Formel:

$$
I_N=\frac{I}{\text{Max}} \Rightarrow I_N=\frac{I}{255}
$$

Zur Normalisierung der Daten, werden also alle Pixelwerte durch 255 dividiert, bevor das Bild dem neuronalen Netz gezeigt wird.

### Datenmehrung

Je mehr Trainingsdaten für das KNN vorhanden sind, desto besser kann es auch mit ungesehenen Daten umgehen. Da nur begrenzt viele Daten zur Verfügung stehen werden in dieser Arbeit einige Methoden der künstlichen  Datenvermehrung angewandt. Dazu werden die Bilder der Eingangsdaten zum Beispiel gespiegelt, verzerrt oder gedreht. Durch die Spiegelung eines Bildes entsteht wieder ein neues Bild, welches zum Training des neuronalen Netzes verwendet werden kann. So kann mit dieser relativ einfachen Methode die Anzahl der Trainingsdaten sehr einfach verdoppelt werden. Die Methode die zum spiegeln der Bilder des *FER+* Datensatzes verwendet wurde ist in LISTING X TODO: zu sehen.

TODO: Code Listing spiegeln

## Entwurf neuronaler Netze

### Topoligien

### Trainingsmethoden


## Evaluierung neuronaler Netze?

### abschnitt 1

### abschnitt 2

## Entwicklung eines Webservice




<!-- expose:
3. Umsetzung & Evaluierung
	1. Prototyping
		1. Aufbau verschiedener neuronaler Netze
		2. Trainieren der Modelle
		3. Bereitstellung als Webservice
	2. Experiment
		1. Verifizieren mit Testdaten
		2. Untersuchung der Genauigkeit
	3. Ergebnisse
-->